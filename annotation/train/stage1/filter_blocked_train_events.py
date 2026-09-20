#!/usr/bin/env python3
"""Remove preview/intro/combination/medley-mix events from train_events JSON files."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

KEEP_LABEL_SUBSTR = ("mixed beeswax soup",)
BLOCK_LABEL_SUBSTR = ("preview", "intro", "combination")


def is_blocked_label(label: str) -> bool:
    ll = (label or "").lower()
    if any(k in ll for k in KEEP_LABEL_SUBSTR):
        return False
    if any(kw in ll for kw in BLOCK_LABEL_SUBSTR):
        return True
    if "mix" in ll and "mixing" not in ll:
        return True
    return False


def filter_events(events: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for event in events:
        label = str(event.get("label") or event.get("seg_label") or "")
        if is_blocked_label(label):
            removed.append(event)
        else:
            kept.append(event)
    return kept, removed


def filter_file(path: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events")
    if not isinstance(events, list):
        raise ValueError(f"{path} 缺少 events 数组")
    kept, removed = filter_events(events)
    if not dry_run:
        data["events"] = kept
        if "event_count" in data:
            data["event_count"] = len(kept)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "before": len(events),
        "after": len(kept),
        "removed": len(removed),
        "removed_labels": sorted({e.get("label", "") for e in removed}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter blocked labels from train_events JSON")
    ap.add_argument("paths", nargs="+", type=Path, help="JSON files to filter in place")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", type=Path, default=None, help="optional removal log JSON")
    args = ap.parse_args()

    reports = [filter_file(p, dry_run=args.dry_run) for p in args.paths]
    for r in reports:
        print(
            f"{r['path']}: {r['before']} -> {r['after']} "
            f"(removed {r['removed']}) labels={r['removed_labels']}"
        )

    if args.log and not args.dry_run:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "rules": [
                "preview / intro / combination: substring on label or seg_label",
                "mix: substring, exclude 'mixing'; keep 'mixed beeswax soup'",
            ],
            "reports": reports,
        }
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
