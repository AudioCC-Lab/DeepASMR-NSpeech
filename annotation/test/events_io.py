"""读写 test_v5.json（data 数组，wav/seg_label 格式）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def seg_label(event: dict) -> str:
    return (event.get("seg_label") or event.get("label") or "").strip()


def event_key(event: dict) -> str:
    wav = (event.get("wav") or "").strip()
    if wav:
        return Path(wav).stem
    return seg_label(event) or "unknown"


def load_test_events(path: Path) -> Tuple[Dict[str, Any], List[dict]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} 需为 {{data: [...]}} 对象")
    data = obj.get("data")
    if not isinstance(data, list):
        raise ValueError(f"{path} 缺少 data 数组")
    return obj, [dict(e) for e in data if isinstance(e, dict)]
