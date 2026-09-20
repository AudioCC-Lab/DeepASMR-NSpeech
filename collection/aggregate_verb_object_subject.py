"""
从若干 audio 根路径下递归收集所有 segment_label_parse.json，
统计 segments 中出现过的 verb / object / subject 取值频次，
写入 root_dir 下的 raw_verb.txt、raw_object.txt、raw_subject.txt（制表符分隔：取值<TAB>计数，按计数降序）。

跳过 null、空串与仅空白。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


def iter_segment_label_parse_files(audio_roots: Iterable[Path]) -> Iterator[Path]:
    for root in audio_roots:
        p = root.resolve()
        if not p.exists():
            continue
        if p.is_file() and p.name == "segment_label_parse.json":
            yield p
            continue
        if p.is_dir():
            yield from p.rglob("segment_label_parse.json")


def normalize_token(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    s = str(v).strip()
    return s if s else None


def collect_from_json(path: Path, verbs: Counter[str], objects: Counter[str], subjects: Counter[str]) -> None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    segs = payload.get("segments")
    if not isinstance(segs, list):
        return
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        vt = normalize_token(seg.get("verb"))
        if vt is not None:
            verbs[vt] += 1
        ot = normalize_token(seg.get("object"))
        if ot is not None:
            objects[ot] += 1
        st = normalize_token(seg.get("subject"))
        if st is not None:
            subjects[st] += 1


def write_counter_txt(out_path: Path, ctr: Counter[str]) -> None:
    lines = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0].casefold()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for label, n in lines:
            f.write(f"{label}\t{n}\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="汇总 segment_label_parse.json 中的 verb/object/subject 取值计数，写入 raw_*.txt",
    )
    ap.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="输出目录（写入 raw_verb.txt / raw_object.txt / raw_subject.txt）",
    )
    ap.add_argument(
        "--audio-path",
        type=Path,
        action="append",
        default=[],
        metavar="DIR_OR_FILE",
        help="audio 根目录或单个 segment_label_parse.json；可重复传入多条",
    )
    args = ap.parse_args()

    roots = [p for p in (args.audio_path or []) if p is not None]
    if not roots:
        print("[ERR] 至少传入一条 --audio-path", file=sys.stderr)
        return 2

    out_root = args.root_dir.resolve()

    verbs: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    subjects: Counter[str] = Counter()

    seen_files: set[Path] = set()
    n_files = 0
    for jp in iter_segment_label_parse_files(roots):
        rp = jp.resolve()
        if rp in seen_files:
            continue
        seen_files.add(rp)
        collect_from_json(rp, verbs, objects, subjects)
        n_files += 1

    write_counter_txt(out_root / "raw_verb.txt", verbs)
    write_counter_txt(out_root / "raw_object.txt", objects)
    write_counter_txt(out_root / "raw_subject.txt", subjects)

    print(
        f"完成: 读取 segment_label_parse.json × {n_files}, "
        f"写入 {out_root / 'raw_verb.txt'}, {out_root / 'raw_object.txt'}, {out_root / 'raw_subject.txt'}"
    )
    print(f"  verb={len(verbs)} 条 object={len(objects)} 条 subject={len(subjects)} 条（不同取值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
