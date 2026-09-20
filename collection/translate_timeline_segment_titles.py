"""
将 timeline.csv 中 segment_title 含日文的部分译为英文（就地改写）。

用法:
  pip install deep-translator
  python collection/translate_timeline_segment_titles.py --dir data/audio

可选 --dry-run 只看统计不写文件；--delay 控制请求间隔降低限流风险。
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepasmr_nspeech import RunSummary  # noqa: E402


def has_japanese(text: str) -> bool:
    if not text or not text.strip():
        return False
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x309F:  # Hiragana
            return True
        if 0x30A0 <= cp <= 0x30FF:  # Katakana
            return True
        if 0x4E00 <= cp <= 0x9FFF:  # CJK Unified (含日文汉字)
            return True
    return False


def translate_batch(
    unique_texts: list[str],
    *,
    delay_sec: float,
    chunk_size: int = 35,
    run_summary: RunSummary,
) -> dict[str, str]:
    try:
        from deep_translator import GoogleTranslator
    except ImportError as e:
        raise SystemExit("请先安装: pip install deep-translator") from e

    translator = GoogleTranslator(source="auto", target="en")
    out: dict[str, str] = {}
    n = len(unique_texts)
    done = 0
    for i in range(0, n, chunk_size):
        chunk = unique_texts[i : i + chunk_size]
        try:
            translated = translator.translate_batch(chunk)
            if not translated or len(translated) != len(chunk):
                raise RuntimeError(f"batch 长度异常: {len(translated or [])} != {len(chunk)}")
            for src, dst in zip(chunk, translated):
                out[src] = (dst.strip() if dst else "") or src
                run_summary.add("success", item_id=src)
        except Exception as ex:
            print(f"[WARN] 批量翻译失败，改为逐条: {ex}", file=sys.stderr)
            for text in chunk:
                try:
                    en = translator.translate(text)
                    out[text] = (en.strip() if en else "") or text
                    run_summary.add("success", item_id=text)
                except Exception as ex2:
                    print(f"[WARN] 翻译失败，保留原文: {text!r} ({ex2})", file=sys.stderr)
                    out[text] = text
                    run_summary.add("failed", item_id=text, detail=ex2)
                if delay_sec > 0:
                    time.sleep(delay_sec)
        done += len(chunk)
        print(f"  翻译进度 {done}/{n}")
        if delay_sec > 0 and i + chunk_size < n:
            time.sleep(delay_sec)
    return out


def process_file(path: Path, mapping: dict[str, str], *, dry_run: bool) -> tuple[int, bool]:
    """返回 (本文件替换的行数, 是否有改动)。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not rows:
        return 0, False
    if fieldnames is None:
        fieldnames = ["video_id", "time", "seconds", "segment_title"]

    replaced = 0
    new_rows: list[dict[str, str]] = []
    changed = False
    for row in rows:
        title = row.get("segment_title") or ""
        if title in mapping and mapping[title] != title:
            row = dict(row)
            row["segment_title"] = mapping[title]
            replaced += 1
            changed = True
        new_rows.append(row)

    if changed and not dry_run:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(new_rows)

    return replaced, changed


def main() -> int:
    ap = argparse.ArgumentParser(description="把 timeline.csv 里日文 segment_title 译为英文")
    ap.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="含各视频 id 子目录的根目录（例如 tabi_audio）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只统计与预览翻译，不写 CSV")
    ap.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="每批翻译后的休眠秒数，减轻限流（默认 0.25）",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=35,
        help="translate_batch 每批条数（默认 35）",
    )
    args = ap.parse_args()

    root: Path = args.dir.resolve()
    if not root.is_dir():
        print(f"[ERR] 不是目录: {root}", file=sys.stderr)
        return 1

    timelines = sorted(root.glob("*/timeline.csv"))
    if not timelines:
        print(f"[ERR] 未找到 */timeline.csv: {root}", file=sys.stderr)
        return 1

    need_translate: set[str] = set()
    for tl in timelines:
        try:
            with tl.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    t = (row.get("segment_title") or "").strip()
                    if t and has_japanese(t):
                        need_translate.add(t)
        except OSError as e:
            print(f"[WARN] 跳过 {tl}: {e}", file=sys.stderr)

    unique = sorted(need_translate)
    run_summary = RunSummary("collection.translate_timeline_labels")
    print(f"发现含日文 segment_title 的唯一文案数: {len(unique)}（timeline 文件数: {len(timelines)}）")
    if not unique:
        print("无需翻译。")
        run_summary.finish(root / "translation_run_summary.json")
        return 0

    if args.dry_run:
        print("\n[dry-run] 含日文 segment_title 示例（最多 20 条，不调用翻译接口）:")
        for k in unique[:20]:
            print(f"  {k!r}")
        for text in unique:
            run_summary.add("skipped", item_id=text, reason="dry_run")
        run_summary.finish(root / "translation_run_summary.json")
        return 0

    mapping: dict[str, str] = {s: s for s in unique}
    mapping.update(
        translate_batch(
            unique,
            delay_sec=max(0.0, float(args.delay)),
            chunk_size=max(1, int(args.chunk_size)),
            run_summary=run_summary,
        )
    )

    total_replaced = 0
    files_changed = 0
    for tl in timelines:
        n_rep, chg = process_file(tl, mapping, dry_run=False)
        if chg:
            files_changed += 1
            total_replaced += n_rep

    print(f"完成：改写 {files_changed} 个 timeline.csv，共替换 {total_replaced} 条 segment_title（已写 .bak）")
    run_summary.finish(
        root / "translation_run_summary.json",
        metadata={"timeline_files": len(timelines), "files_changed": files_changed},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
