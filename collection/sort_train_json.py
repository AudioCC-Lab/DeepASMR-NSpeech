"""
将 train.json 中相同 video id 的条目聚合在一起，并按 start_time、end_time 升序排列。

用法:
  python sort_train_json.py --train-json /path/to/train.json
  python sort_train_json.py --train-json /path/to/train.json --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_code_dir = Path(__file__).resolve().parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

from timeline_parse import parse_seconds_from_timestamp_token


def _time_sort_key(entry: dict) -> tuple[int, int, str]:
    start = parse_seconds_from_timestamp_token(entry.get("start_time") or "") or 0
    end = parse_seconds_from_timestamp_token(entry.get("end_time") or "") or 0
    return (start, end, entry.get("audio_path") or "")


def sort_train_entries(entries: list[dict]) -> list[dict]:
    """按 id 分组，组内按 start/end 时间排序；组间按 id 字典序。"""
    by_id: dict[str, list[dict]] = {}
    for row in entries:
        vid = str(row.get("id") or "").strip()
        if not vid:
            raise ValueError(f"条目缺少 id: {row!r}")
        by_id.setdefault(vid, []).append(row)

    out: list[dict] = []
    for vid in sorted(by_id.keys()):
        block = sorted(by_id[vid], key=_time_sort_key)
        out.extend(block)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="按 video id 聚合 train.json，组内按 start/end 时间排序")
    ap.add_argument("--train-json", type=Path, required=True, help="输入 train.json 路径")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出路径（默认 <train-json>.sorted.json；--in-place 时忽略）",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="原地覆盖 train-json（会先备份到 _backup_before_sort_train/<timestamp>/）",
    )
    args = ap.parse_args()

    train_path = args.train_json.resolve()
    if not train_path.is_file():
        raise SystemExit(f"文件不存在: {train_path}")

    raw = train_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise SystemExit("train.json 顶层必须是 JSON 数组")

    sorted_data = sort_train_entries(data)
    if sorted_data == data:
        print(f"[OK] 已是正确顺序，无需改写: {train_path}")
        return 0

    out_text = json.dumps(sorted_data, ensure_ascii=False, indent=2) + "\n"

    if args.in_place:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = train_path.parent / "_backup_before_sort_train" / ts
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / train_path.name
        shutil.copy2(train_path, backup_path)
        train_path.write_text(out_text, encoding="utf-8")
        print(f"[OK] 已排序并覆盖: {train_path}")
        print(f"  备份: {backup_path}")
    else:
        out_path = (args.output or train_path.with_suffix(".sorted.json")).resolve()
        out_path.write_text(out_text, encoding="utf-8")
        print(f"[OK] 已写入: {out_path} ({len(sorted_data)} 条)")

    n_ids = len({str(x.get("id") or "") for x in sorted_data})
    print(f"  视频 id 数: {n_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
