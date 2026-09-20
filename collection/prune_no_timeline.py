"""
删除「在 <audio-dir>/<id>/ 下存在目录、但没有有效 timeline.csv」的条目（默认 audio-dir = <root>/audio）：
- 删除整个该 id 目录（含音频、frames、半成品等）
- 从 meta.json 去掉对应 id
- 从 export.csv 去掉能解析到该 id 的所有行

判定「有效 timeline」与 list_audio_without_timeline.has_timeline_rows 一致。
默认只打印统计；加 --apply 才执行，并先备份 export.csv / meta.json。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import shutil
from pathlib import Path

from dedupe_ids import extract_id_from_row
from list_audio_without_timeline import has_timeline_rows


def _backup_file(src: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / src.name
    shutil.copy2(src, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(
        description="清理无有效 timeline 的 audio 子目录，并同步移除 export.csv / meta.json 中对应记录",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="元数据根目录（含 export.csv、meta.json；备份与 pruned_*.txt 写在此）",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="各视频 id 子目录的父目录（默认：<root>/audio），须与下载 --audio-dir 一致",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="执行删除与写回；不加则仅报告将影响的 id 与行数",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="可选：将 JSON 报告写入该路径",
    )
    args = ap.parse_args()
    root = args.root.resolve()
    audio_dir = (args.audio_dir or (root / "audio")).resolve()
    csv_path = root / "export.csv"
    meta_path = root / "meta.json"

    if not audio_dir.is_dir():
        raise SystemExit(f"audio 目录不存在: {audio_dir}")
    if not csv_path.is_file():
        raise SystemExit(f"export.csv 不存在: {csv_path}")
    if not meta_path.is_file():
        raise SystemExit(f"meta.json 不存在: {meta_path}")

    prune_ids: list[str] = []
    for entry in sorted(audio_dir.iterdir(), key=lambda p: p.name.casefold()):
        if not entry.is_dir():
            continue
        vid = entry.name
        if not has_timeline_rows(entry / "timeline.csv"):
            prune_ids.append(vid)

    prune_set = frozenset(prune_ids)
    if not prune_ids:
        print("[OK] 没有需要清理的 id（所有 audio 子目录均有有效 timeline.csv）")
        return 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    kept_rows = [r for r in rows if extract_id_from_row(r) not in prune_set]
    removed_csv = len(rows) - len(kept_rows)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise SystemExit("meta.json 顶层必须是 JSON 对象（按视频 id 为键）")

    removed_meta = [k for k in prune_ids if k in meta]
    new_meta = {k: v for k, v in meta.items() if k not in prune_set}

    report = {
        "root": str(root),
        "audio_dir": str(audio_dir),
        "prune_ids": prune_ids,
        "export_rows_before": len(rows),
        "export_rows_removed": removed_csv,
        "export_rows_after": len(kept_rows),
        "meta_keys_removed": removed_meta,
        "apply": bool(args.apply),
    }

    print(f"[PLAN] 将清理无有效 timeline 的 id 共 {len(prune_ids)} 个：")
    for vid in prune_ids:
        print(f"  - {vid}")
    print(f"[PLAN] export.csv 将删除 {removed_csv} 行，保留 {len(kept_rows)} 行")
    print(f"[PLAN] meta.json 将删除 {len(removed_meta)} 个键")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] 报告已写入 {args.report}")

    if not args.apply:
        print("[INFO] 未执行删除。确认无误后追加 --apply")
        return 0

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = root / "_backup_before_prune_no_timeline" / ts
    _backup_file(csv_path, backup_dir)
    _backup_file(meta_path, backup_dir)
    print(f"[OK] 已备份到 {backup_dir}")

    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept_rows)
    tmp_csv.replace(csv_path)
    print(f"[OK] 已写回 {csv_path.name}")

    tmp_meta = meta_path.with_suffix(".json.tmp")
    with tmp_meta.open("w", encoding="utf-8") as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)
    tmp_meta.replace(meta_path)
    print(f"[OK] 已写回 {meta_path.name}")

    for vid in prune_ids:
        target = audio_dir / vid
        if target.is_dir():
            shutil.rmtree(target)
            try:
                disp = target.relative_to(root)
            except ValueError:
                disp = target
            print(f"[OK] 已删除目录 {disp}")

    audit_path = root / "pruned_no_timeline_ids.txt"
    audit_path.write_text("\n".join(prune_ids) + "\n", encoding="utf-8")
    print(f"[OK] 本次清理的 id 列表已写入 {audit_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
