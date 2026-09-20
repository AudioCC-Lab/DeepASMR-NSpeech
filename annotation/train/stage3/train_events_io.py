"""读写 train_events_with_visual.json（event 级增量）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def event_key(event: dict) -> str:
    return f"{event['id']}_{int(float(event['start_sec']))}_{int(float(event['end_sec']))}"


def load_train_events(path: Path) -> Tuple[Dict[str, Any], List[dict]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} 需为 {{events: [...]}} 对象")
    events = obj.get("events")
    if not isinstance(events, list):
        raise ValueError(f"{path} 缺少 events 数组")
    return obj, [dict(e) for e in events if isinstance(e, dict)]


def representative_audio_path(event: dict) -> Optional[Path]:
    for seg in event.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        raw = (seg.get("audio_path") or "").strip()
        if not raw:
            continue
        p = Path(raw)
        if p.is_file():
            return p
    segs = event.get("segments") or []
    if segs and isinstance(segs[0], dict):
        raw = (segs[0].get("audio_path") or "").strip()
        if raw:
            return Path(raw)
    return None
