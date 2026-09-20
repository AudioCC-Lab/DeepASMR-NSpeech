#!/usr/bin/env python3
"""Select SVO-AQA questions that pass reference-versus-silence voting."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def load_run(path: Path) -> tuple[dict, Dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a result list or contain results")
    return payload.get("meta") or {}, {
        str(row["question_id"]): row for row in rows if row.get("question_id")
    }


def predicted(row: dict) -> str | None:
    value = row.get("pred_logprob") or row.get("pred_label") or row.get("pred")
    return str(value).strip().upper() if value else None


def is_correct(row: dict) -> bool:
    gold = row.get("gold_label")
    return predicted(row) == (str(gold).strip().upper() if gold else None)


def task_accuracy(runs: List[Dict[str, dict]], ids: Iterable[str]) -> Dict[str, float]:
    hits: Counter = Counter()
    totals: Counter = Counter()
    for run in runs:
        for question_id in ids:
            row = run[question_id]
            task = str(row.get("task") or "unknown")
            totals[task] += 1
            hits[task] += int(is_correct(row))
    return {task: hits[task] / totals[task] for task in sorted(totals)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, action="append", required=True)
    parser.add_argument("--silence-run", type=Path, action="append", required=True)
    parser.add_argument("--reference-min-votes", type=int, default=2)
    parser.add_argument("--silence-max-votes", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.reference_run) != len(args.silence_run):
        raise SystemExit("reference and silence run counts must match")

    reference_loaded = [load_run(path) for path in args.reference_run]
    silence_loaded = [load_run(path) for path in args.silence_run]
    reference = [rows for _, rows in reference_loaded]
    silence = [rows for _, rows in silence_loaded]
    common = set.intersection(*(set(run) for run in reference + silence))

    selected: List[dict] = []
    for question_id in sorted(common):
        ref_votes = sum(is_correct(run[question_id]) for run in reference)
        silence_votes = sum(is_correct(run[question_id]) for run in silence)
        if ref_votes < args.reference_min_votes or silence_votes > args.silence_max_votes:
            continue
        source = reference[0][question_id]
        selected.append(
            {
                "question_id": question_id,
                "audio_id": source.get("audio_id"),
                "task": source.get("task"),
                "reference_correct_votes": ref_votes,
                "silence_correct_votes": silence_votes,
            }
        )

    ids = [row["question_id"] for row in selected]
    ref_by_task = task_accuracy(reference, ids)
    silence_by_task = task_accuracy(silence, ids)
    payload = {
        "meta": {
            "selection": (
                f"reference_correct_votes >= {args.reference_min_votes} of {len(reference)} "
                f"and silence_correct_votes <= {args.silence_max_votes} of {len(silence)}"
            ),
            "reference_judge_models": sorted(
                {str(meta.get("judge_model") or meta.get("omni_model") or "unknown") for meta, _ in reference_loaded}
            ),
            "silence_judge_models": sorted(
                {str(meta.get("judge_model") or meta.get("omni_model") or "unknown") for meta, _ in silence_loaded}
            ),
            "n_questions": len(selected),
            "n_clips": len({row.get("audio_id") for row in selected}),
            "n_by_task": dict(Counter(str(row.get("task")) for row in selected)),
            "reference_accuracy_by_task": ref_by_task,
            "silence_accuracy_by_task": silence_by_task,
            "reference_macro_accuracy": sum(ref_by_task.values()) / len(ref_by_task) if ref_by_task else None,
            "silence_macro_accuracy": sum(silence_by_task.values()) / len(silence_by_task) if silence_by_task else None,
        },
        "questions": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["meta"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
