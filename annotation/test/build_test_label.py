#!/usr/bin/env python3
"""Export normalized test annotations without private quality-evaluation data.

The input is the output of ``run_svo_normalize.py``. Rows with failed verb
annotation are excluded by default and duplicate ``(subject, verb, object,
seg_label)`` rows are collapsed. Output metadata records operational skip
counts; no private human ground truth is required.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LIB = HERE / "lib"
for path in (LIB, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from annotation.common.info_common import (  # noqa: E402
    atomic_write_json,
    load_info_full_payload,
    load_verb_class_lexicon,
)
from deepasmr_nspeech import RunSummary  # noqa: E402

DEFAULT_INPUT = HERE / "outputs" / "seg_label_svo_normalized.json"
DEFAULT_OUTPUT = HERE / "outputs" / "test_label.json"
DEFAULT_VERB_CLASSES = REPO_ROOT / "vocab" / "asmr_verb_classes_en.csv"

PUBLIC_FIELDS = ("wav", "caption", "verb", "verb_class", "subject", "object", "creator_id")
TRACE_FIELDS = ("seg_label", "title", "visual_description", "verb_source", "noun_source")


def dedup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("subject"),
        row.get("verb"),
        row.get("object"),
        (row.get("seg_label") or "").strip(),
    )


def export_rows(
    rows: list[dict[str, Any]],
    *,
    verb_classes: Path,
    include_failed: bool,
    include_trace_fields: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lex = load_verb_class_lexicon(verb_classes)
    seen: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for row in rows:
        if not include_failed and row.get("status") != "ok":
            skipped["status_not_ok"] += 1
            continue
        if not row.get("verb"):
            skipped["missing_verb"] += 1
            continue
        if not include_failed and row.get("verb_failed"):
            skipped["verb_failed"] += 1
            continue

        key = dedup_key(row)
        if key in seen:
            skipped["duplicate_svo_and_segment_label"] += 1
            continue
        seen.add(key)

        exported = {field: row.get(field) for field in PUBLIC_FIELDS}
        if not exported.get("verb_class"):
            exported["verb_class"] = lex.class_of(str(exported.get("verb") or ""))
        if include_trace_fields:
            exported.update({field: row.get(field) for field in TRACE_FIELDS})
        output.append(exported)

    stats = {
        "input": len(rows),
        "success": len(output),
        "failed_or_skipped": sum(skipped.values()),
        "skip_reasons": dict(sorted(skipped.items())),
        "unique_svo": len({(r.get("subject"), r.get("verb"), r.get("object")) for r in output}),
        "unique_svo_and_segment_label": len(seen),
    }
    return output, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verb-classes", type=Path, default=DEFAULT_VERB_CLASSES)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument(
        "--include-trace-fields",
        action="store_true",
        help="Include source timeline/title/visual fields for local auditing. Disabled for public labels.",
    )
    args = parser.parse_args()

    _, rows = load_info_full_payload(args.input)
    exported, stats = export_rows(
        rows,
        verb_classes=args.verb_classes,
        include_failed=args.include_failed,
        include_trace_fields=args.include_trace_fields,
    )
    payload = {
        "meta": {
            "dataset": "DeepASMR-NSpeech",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **stats,
        },
        "data": exported,
    }
    atomic_write_json(args.output, payload)
    run_summary = RunSummary("annotation.test.export_labels")
    run_summary.status_counts["success"] = stats["success"]
    run_summary.status_counts["skipped"] = stats["failed_or_skipped"]
    run_summary.reason_counts.update(stats["skip_reasons"])
    run_summary.finish(args.output.with_suffix(".run_summary.json"))
    print(
        f"[DONE] input={stats['input']} success={stats['success']} "
        f"failed_or_skipped={stats['failed_or_skipped']} reasons={stats['skip_reasons']} "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
