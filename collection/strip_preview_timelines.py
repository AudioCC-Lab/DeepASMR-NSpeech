"""
扫描 <audio-dir>/*/timeline.csv（默认 <root>/audio）：删除 segment_title 命中任一指定关键词的行（子串、不区分大小写），
再按 normalize_timeline_csv 同一套逻辑重算 start/end 并写回。

说明：匹配为子串；关键词出现在其它词中间时也会命中，请自行控制列表。

用法：
  python strip_preview_timelines.py --root /path/to/data
  python strip_preview_timelines.py --root /path/to/data --keywords preview,intro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import csv
from normalize_timeline_csv import REQUIRED_OLD, normalize_rows_to_disk


def parse_keywords_arg(s: str) -> tuple[list[str], str]:
    """返回 (小写关键词列表, 用于展示的原文拼接串)。"""
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if not parts:
        raise SystemExit("至少需要 1 个关键词，例如 --keywords preview 或 preview,intro")
    lower = [p.lower() for p in parts]
    return lower, ", ".join(parts)


def _title_matches_keywords(segment_title: str, keywords_lc: list[str]) -> bool:
    t = (segment_title or "").lower()
    return any(kw in t for kw in keywords_lc)


def process_file(
    path: Path,
    *,
    keywords_lc: list[str],
    dry_run: bool,
) -> tuple[bool, str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if not text.strip():
        return False, "empty file"

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return False, "no header"
        fn = {h.strip(): h for h in reader.fieldnames if h}
        missing = REQUIRED_OLD - set(fn.keys())
        if missing:
            return False, f"missing columns {sorted(missing)}"

        norm_rows_raw: list[dict[str, str]] = []
        removed = 0
        for row in reader:
            if not isinstance(row, dict):
                continue
            nr = {(k or "").strip(): (row.get(fn.get(k, k)) or "").strip() for k in REQUIRED_OLD}
            if _title_matches_keywords(nr.get("segment_title", ""), keywords_lc):
                removed += 1
                continue
            norm_rows_raw.append(
                {
                    "video_id": nr.get("video_id", ""),
                    "time": nr.get("time", ""),
                    "seconds": nr.get("seconds", ""),
                    "segment_title": nr.get("segment_title", ""),
                }
            )

    if removed == 0:
        return True, "no rows matched keywords"

    if not norm_rows_raw:
        return (
            False,
            f"removed {removed} matched row(s); no rows left, refuse to write empty",
        )

    prefix = f"removed {removed} matched row(s); "
    return normalize_rows_to_disk(path, norm_rows_raw, dry_run=dry_run, msg_prefix=prefix)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从各 <audio-dir>/<id>/timeline.csv 去掉 segment_title 命中关键词的行并重新标准化 start/end",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="元数据根目录（用于默认 <root>/audio；可与下载 --root 对齐）",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="各视频 id 子目录的父目录（默认：<root>/audio），须与下载 --audio-dir 一致",
    )
    ap.add_argument(
        "--keywords",
        type=str,
        default="preview",
        help='逗号分隔多个词；任一在 segment_title 中子串匹配（忽略大小写）即删该行。默认 "preview"',
    )
    ap.add_argument("--dry-run", action="store_true", help="只演练，不写回文件")
    args = ap.parse_args()
    keywords_lc, keywords_show = parse_keywords_arg(args.keywords)
    root = args.root.resolve()
    audio = (args.audio_dir or (root / "audio")).resolve()
    if not audio.is_dir():
        print(f"audio 目录不存在: {audio}", file=sys.stderr)
        return 2

    files = sorted(audio.glob("*/timeline.csv"), key=lambda p: str(p.parent.name).casefold())
    ok, bad = 0, 0
    for p in files:
        rel = p.relative_to(audio)
        good, msg = process_file(p, keywords_lc=keywords_lc, dry_run=args.dry_run)
        tag = "[OK]" if good else "[SKIP]"
        print(f"{tag} {rel}: {msg}")
        if good:
            ok += 1
        else:
            bad += 1
    suf = " (dry-run)" if args.dry_run else ""
    print(f"[SUMMARY]{suf} 关键词 [{keywords_show}] → 成功/无改动 {ok}, 跳过/失败 {bad}, 合计 {len(files)}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
