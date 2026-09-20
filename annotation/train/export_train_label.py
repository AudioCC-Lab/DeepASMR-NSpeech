#!/usr/bin/env python3
"""Export annotated train events to clip-level DeepASMR-NSpeech labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation.common.info_common import atomic_write_json
from deepasmr_nspeech import RunSummary

KEEP_KEYS = ("wav", "seg_label", "title", "verb", "object", "subject", "author", "caption")


def build_caption(event: dict) -> str:
    caption = (event.get("caption") or "").strip()
    if caption:
        return caption
    parts = []
    for key in ("subject", "verb", "object"):
        val = event.get(key)
        if val is not None and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def export_train_label(src: Path, out: Path) -> dict:
    with src.open(encoding="utf-8") as f:
        wrapper = json.load(f)

    events = wrapper.get("events") or []
    rows = []
    seen_wav: set[str] = set()
    summary = RunSummary("annotation.train.export_labels")
    stats = {
        "n_events": len(events),
        "skipped_no_verb": 0,
        "skipped_dup_wav": 0,
        "skipped_no_audio": 0,
    }

    for event in events:
        event_id = str(event.get("id") or event.get("label") or "unknown")
        verb = (event.get("verb") or "").strip()
        if not verb:
            stats["skipped_no_verb"] += 1
            summary.add("skipped", item_id=event_id, reason="missing_verb")
            continue

        caption = build_caption(event)
        row_base = {
            "seg_label": (event.get("label") or "").strip(),
            "title": (event.get("title") or "").strip(),
            "verb": verb,
            "object": event.get("object") if event.get("object") is not None else "",
            "subject": event.get("subject") if event.get("subject") is not None else "",
            "author": (event.get("author") or "").strip(),
            "caption": caption,
        }

        segments = event.get("segments") or []
        if not segments:
            summary.add("skipped", item_id=event_id, reason="missing_segments")
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            wav = (seg.get("audio_path") or "").strip()
            if not wav:
                stats["skipped_no_audio"] += 1
                summary.add("skipped", item_id=event_id, reason="missing_audio_path")
                continue
            if wav in seen_wav:
                stats["skipped_dup_wav"] += 1
                summary.add("skipped", item_id=wav, reason="duplicate_audio_path")
                continue
            seen_wav.add(wav)
            rows.append({"wav": wav, **row_base})
            summary.add("success", item_id=wav)

    payload = {"data": rows}
    atomic_write_json(out, payload)
    summary.finish(out.with_suffix(".run_summary.json"))

    stats["n_rows"] = len(rows)
    stats["out_json"] = str(out)
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出 train_label.json")
    p.add_argument(
        "--src-json",
        type=Path,
        default=Path(__file__).resolve().parent / "train_events_with_visual.json",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "train_label.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.src_json.is_file():
        print(f"[ERROR] 未找到 {args.src_json}")
        return 1
    stats = export_train_label(args.src_json, args.out_json)
    print(
        f"[DONE] events={stats['n_events']} rows={stats['n_rows']} "
        f"skip_no_verb={stats['skipped_no_verb']} -> {stats['out_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
