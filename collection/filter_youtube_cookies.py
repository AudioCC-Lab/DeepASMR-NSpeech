"""
从 Netscape cookies.txt 中筛出 YouTube / Google 相关条目，写入新文件。

用法:
  python filter_youtube_cookies.py --input /path/to/cookies_all.txt
  python filter_youtube_cookies.py --input /path/to/cookies_all.txt --output /secure/path/cookies_youtube.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_code_dir = Path(__file__).resolve().parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

from download_yt_audio_and_timeline import write_youtube_cookies_filtered  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="筛选 cookies.txt 中 YouTube/Google 相关条目")
    ap.add_argument("--input", type=Path, required=True, help="原始 Netscape cookies.txt")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出路径（默认与 input 同目录下的 cookies_youtube.txt）",
    )
    args = ap.parse_args()

    src = args.input.resolve()
    if not src.is_file():
        raise SystemExit(f"输入文件不存在: {src}")

    dst = (args.output or (src.parent / "cookies_youtube.txt")).resolve()
    kept, skipped = write_youtube_cookies_filtered(src, dst)

    print(f"[OK] 已写入: {dst}")
    print(f"  保留 YouTube/Google cookie 行: {kept}")
    if skipped:
        print(f"  跳过非法行: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
