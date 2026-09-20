"""
segments/<id>_<cleanStart>_<cleanEnd>/ 目录约定（与 export_clean_audio_segments 一致）。"""
from __future__ import annotations

from typing import Tuple

INFO_JSON = "info.json"
AUDIO_WAV = "audio.wav"
KEYFRAME_PNG = "keyframe.png"


def parse_clean_start_clean_end_field(s: str) -> Tuple[float, float]:
    """解析 info.json 内 clean_start_clean_end 字段（形如 40_248 或 12p5_90p3）。"""
    t = (s or "").strip()
    if "_" not in t:
        raise ValueError(f"clean_start_clean_end 无效: {s!r}")
    a, b = t.split("_", 1)

    def tok(x: str) -> float:
        x = x.strip()
        if not x:
            raise ValueError("empty token")
        return float(x.replace("p", "."))

    return tok(a), tok(b)
