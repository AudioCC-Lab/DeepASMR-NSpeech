#!/usr/bin/env python3
"""
Initialize test_v5.json from prepared segment context.

用法:
  python3 annotation/test/init_test_v5.py --context-json /path/to/context.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepasmr_nspeech import RunSummary

ROOT = Path(__file__).resolve().parent
DEFAULT_CONTEXT = ROOT.parent / "info_full_new_with_caption.json"
DEFAULT_OUT = ROOT / "test_v5.json"

KEEP_KEYS = (
    "wav",
    "seg_label",
    "title",
    "author",
    "visual_description",
    "visual_description_model",
)


def is_parallel_seg_label(seg_label: str) -> bool:
    """seg_label 含 + 或 & 表示 up 主标注了两个并行动作，不纳入单 SVO 提取。"""
    s = seg_label or ""
    return "+" in s or "&" in s


def load_rows(path: Path) -> List[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("data") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"{path} 缺少 data 数组")
    return [r for r in rows if isinstance(r, dict)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="初始化 test_v5.json")
    p.add_argument("--context-json", type=Path, default=DEFAULT_CONTEXT)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.context_json.is_file():
        print(f"[ERROR] {args.context_json} 不存在", file=__import__("sys").stderr)
        return 1
    if args.out_json.is_file() and not args.force:
        print(f"[ERROR] {args.out_json} 已存在，加 --force", file=__import__("sys").stderr)
        return 1

    data: List[dict] = []
    summary = RunSummary("annotation.test.initialize")
    for row in load_rows(args.context_json):
        wav = str(row.get("wav") or "").strip()
        if not wav:
            summary.add("skipped", reason="missing_audio_path")
            continue
        seg = (row.get("seg_label") or row.get("label") or "").strip()
        if is_parallel_seg_label(seg):
            summary.add("skipped", item_id=wav, reason="parallel_segment_label")
            continue
        out: Dict[str, Any] = {k: row[k] for k in KEEP_KEYS if k in row}
        data.append(out)
        summary.add("success", item_id=wav)

    payload = {
        "meta": {
            "pipeline": "test_v5",
            "context_json": str(args.context_json),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_rows": len(data),
            "skipped_parallel_seg_label": summary.reason_counts.get("parallel_segment_label", 0),
        },
        "data": data,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary.finish(args.out_json.with_suffix(".run_summary.json"))
    print(f"[OK] {args.out_json} n={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
