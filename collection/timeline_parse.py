"""
YouTube 简介时间戳解析：优先匹配「时:分:秒」，再匹配「分:秒」，
避免出现 2:43:34 被误切成 2:43 + 标题“34 xxx”从而导致秒数偏小。
"""

from __future__ import annotations

import re


# 先试 H:M:S（分、秒各占 1–2 位，与 yt-dlp/简介常见写法一致）
_LINE_HMS_RE = re.compile(
    r"^\s*(?P<t>\d{1,4}:\d{1,2}:\d{2})(?!\d)"
    r"\s*(?:[-–—|:]\s*)?(?P<title>.*)\s*$"
)

# 再试 M:S（整块时间后不能直接跟数字）
_LINE_MS_RE = re.compile(
    r"^\s*(?P<t>\d{1,4}:\d{2})(?!\d)"
    r"\s*(?:[-–—|:]\s*)?(?P<title>.*)\s*$"
)


def parse_seconds_from_timestamp_token(token: str) -> int | None:
    """将 `H:M:S` 或 `M:S` 时间码转为秒（整数）。不合法返回 None。"""
    parts = token.strip().split(":")
    if len(parts) == 2:
        mm, ss = parts
        try:
            return int(mm) * 60 + int(ss)
        except ValueError:
            return None
    if len(parts) == 3:
        hh, mm, ss = parts
        try:
            return int(hh) * 3600 + int(mm) * 60 + int(ss)
        except ValueError:
            return None
    return None


def extract_timestamp_prefix(line: str) -> tuple[str, str] | None:
    """
    解析简介一行：返回 (time_token, title)；无法识别返回 None。
    永远优先尝试 H:M:S，否则会误解析 2:43:34 为 2:43。
    """
    s = line.rstrip("\n")
    m = _LINE_HMS_RE.match(s)
    if m:
        return (m.group("t").strip(), (m.group("title") or "").strip())
    m = _LINE_MS_RE.match(s)
    if m:
        return (m.group("t").strip(), (m.group("title") or "").strip())
    return None
