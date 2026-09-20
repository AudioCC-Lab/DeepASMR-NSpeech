from __future__ import annotations

import argparse
import csv
from pathlib import Path


def has_timeline_rows(timeline_csv: Path) -> bool:
    """存在且至少有一条数据行（非仅表头）则视为有时间轴。"""
    if not timeline_csv.is_file():
        return False
    try:
        with timeline_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return False
            for row in reader:
                if isinstance(row, dict) and any((v or "").strip() for v in row.values()):
                    return True
    except OSError:
        return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="扫描 audio/ 下的各视频目录，列出没有 timeline.csv 或无有效行的视频 id",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="元数据根目录（默认：脚本所在 code 的上级）；与下载 --root 对齐",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="各视频 id 子目录的父目录（默认：<root>/audio）；须与下载 --audio-dir 一致",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出的 txt 路径（默认：<root>/no_timeline_ids.txt）",
    )
    args = ap.parse_args()
    root = args.root.resolve()
    audio_dir = (args.audio_dir or (root / "audio")).resolve()
    out_path = (args.output or (root / "no_timeline_ids.txt")).resolve()

    if not audio_dir.is_dir():
        raise SystemExit(f"audio 目录不存在: {audio_dir}")

    missing: list[str] = []
    for entry in sorted(audio_dir.iterdir(), key=lambda p: p.name.casefold()):
        if not entry.is_dir():
            continue
        vid = entry.name
        if not has_timeline_rows(entry / "timeline.csv"):
            missing.append(vid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    print(f"[OK] 无有效 timeline：{len(missing)} 个 id")
    print(f"      列表已写入：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
