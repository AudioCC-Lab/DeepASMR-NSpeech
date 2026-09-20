import argparse
import csv
import datetime as _dt
import json
import os
import re
import shutil
from collections import Counter, defaultdict


def extract_id_from_row(row: dict) -> str:
    url = (row.get("webpage_url") or "").strip()
    m = re.search(r"[?&]v=([^&]+)", url)
    if m:
        return m.group(1).strip()

    audio_filename = (row.get("audio_filename") or "").strip().replace("\\", "/")
    # expected: audio/<id>/<id>.wav
    m = re.search(r"(?:^|/)audio/([^/]+)/", audio_filename)
    if m:
        return m.group(1).strip()

    return ""


def list_audio_id_groups(audio_dir: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    if not os.path.isdir(audio_dir):
        return groups

    for fn in os.listdir(audio_dir):
        if fn.endswith(".part"):
            continue
        base = fn.split(".", 1)[0]
        if base:
            groups[base].append(fn)

    # stable ordering for determinism
    for k in list(groups.keys()):
        groups[k].sort(key=lambda x: (len(x), x.lower()))
    return dict(groups)


def backup_file(path: str, backup_dir: str) -> str:
    os.makedirs(backup_dir, exist_ok=True)
    dst = os.path.join(backup_dir, os.path.basename(path))
    shutil.copy2(path, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="元数据根目录（含 export.csv、meta.json）；实际使用请显式传入项目根",
    )
    ap.add_argument(
        "--audio-dir",
        default=None,
        help="各视频 id 子目录的父目录（默认：<root>/audio），须与下载 --audio-dir 一致",
    )
    ap.add_argument("--apply", action="store_true", help="Actually delete/overwrite. Default is dry-run.")
    ap.add_argument("--keep", choices=["first", "last"], default="first", help="Which duplicate CSV row to keep per id.")
    ap.add_argument("--report", default="", help="Optional path to write JSON report.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    csv_path = os.path.join(root, "export.csv")
    meta_path = os.path.join(root, "meta.json")
    audio_dir = os.path.abspath(args.audio_dir) if args.audio_dir else os.path.join(root, "audio")

    # --- CSV load ---
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    ids = [extract_id_from_row(r) for r in rows]
    id_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, vid in enumerate(ids):
        if vid:
            id_to_indices[vid].append(idx)

    csv_dups = {k: v for k, v in id_to_indices.items() if len(v) > 1}

    # --- META load ---
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise SystemExit("meta.json must be a JSON object keyed by id.")

    # JSON object keys can't be duplicated; duplicates only possible if id inside value mismatches key.
    meta_key_mismatch = []
    for k, v in meta.items():
        if isinstance(v, dict):
            inner_id = v.get("id") or v.get("video_id")
            if inner_id is not None and str(inner_id).strip() and str(inner_id).strip() != k:
                meta_key_mismatch.append({"key": k, "inner_id": str(inner_id).strip()})

    # --- AUDIO ---
    audio_groups = list_audio_id_groups(audio_dir)
    audio_dups = {k: v for k, v in audio_groups.items() if len(v) > 1}

    # --- Cross-source duplicates: same id referenced multiple times in CSV vs audio/meta presence ---
    csv_id_set = set(id_to_indices.keys())
    meta_id_set = set(meta.keys())
    audio_id_set = set(audio_groups.keys())

    report = {
        "root": root,
        "csv": {
            "path": csv_path,
            "rows": len(rows),
            "fieldnames": fieldnames,
            "unique_ids": len(csv_id_set),
            "duplicate_ids": {k: v for k, v in sorted(csv_dups.items(), key=lambda kv: (-len(kv[1]), kv[0]))},
        },
        "meta": {
            "path": meta_path,
            "entries": len(meta),
            "key_mismatch": meta_key_mismatch,
        },
        "audio": {
            "dir": audio_dir,
            "file_count": sum(len(v) for v in audio_groups.values()),
            "duplicate_ids": {k: v for k, v in sorted(audio_dups.items(), key=lambda kv: (-len(kv[1]), kv[0]))},
        },
        "presence": {
            "ids_in_all_three": sorted(list(csv_id_set & meta_id_set & audio_id_set)),
            "ids_only_in_csv": sorted(list(csv_id_set - (meta_id_set | audio_id_set))),
            "ids_only_in_meta": sorted(list(meta_id_set - (csv_id_set | audio_id_set))),
            "ids_only_in_audio": sorted(list(audio_id_set - (csv_id_set | meta_id_set))),
        },
        "planned_actions": [],
    }

    # --- Apply changes ---
    if args.apply:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(root, "_backup_before_dedupe", ts)
        backup_file(csv_path, backup_dir)
        backup_file(meta_path, backup_dir)

        # CSV dedupe: keep first/last occurrence per id
        keep_index_set = set()
        for vid, idxs in id_to_indices.items():
            if len(idxs) == 1:
                keep_index_set.add(idxs[0])
            else:
                keep_index_set.add(idxs[0] if args.keep == "first" else idxs[-1])

        new_rows = [r for i, r in enumerate(rows) if i in keep_index_set]
        removed_csv_rows = [i for i in range(len(rows)) if i not in keep_index_set]

        tmp_csv = csv_path + ".tmp"
        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        os.replace(tmp_csv, csv_path)

        report["planned_actions"].append(
            {"type": "overwrite_csv_deduped", "kept": len(new_rows), "removed_row_indices": removed_csv_rows}
        )

        # META: no true duplicate keys possible; just optionally fix mismatches by aligning inner id to key.
        if meta_key_mismatch:
            for item in meta_key_mismatch:
                k = item["key"]
                if isinstance(meta.get(k), dict):
                    meta[k]["id"] = k
            tmp_meta = meta_path + ".tmp"
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp_meta, meta_path)
            report["planned_actions"].append(
                {"type": "fix_meta_inner_id_mismatch", "count": len(meta_key_mismatch)}
            )

        # AUDIO: for duplicated id bases, keep best candidate
        deleted_audio_files = []
        for vid, fns in audio_dups.items():
            # prefer real audio extension over extensionless; then prefer larger size.
            candidates = []
            for fn in fns:
                full = os.path.join(audio_dir, fn)
                ext = os.path.splitext(fn)[1].lower()
                if os.path.isdir(full):
                    # Treat per-id folder as best choice if it contains a likely audio file.
                    try:
                        inner = os.listdir(full)
                    except OSError:
                        inner = []
                    inner_audio = any(os.path.splitext(x)[1].lower() in (".wav", ".m4a", ".mp3", ".webm", ".aac", ".flac", ".opus") for x in inner)
                    size = 0
                    for x in inner:
                        xp = os.path.join(full, x)
                        if os.path.isfile(xp):
                            try:
                                size += os.path.getsize(xp)
                            except OSError:
                                pass
                    score_kind = 3 if inner_audio else 1
                    score_ext = 0
                else:
                    score_kind = 2 if os.path.isfile(full) else 0
                    size = os.path.getsize(full) if os.path.exists(full) and os.path.isfile(full) else -1
                    score_ext = 2 if ext in (".wav", ".m4a", ".mp3", ".webm", ".aac", ".flac", ".opus") else (1 if ext else 0)
                candidates.append((score_kind, score_ext, size, -len(fn), fn))
            candidates.sort(reverse=True)
            keep_fn = candidates[0][4]
            for fn in fns:
                if fn == keep_fn:
                    continue
                full = os.path.join(audio_dir, fn)
                if not os.path.exists(full):
                    continue
                if os.path.isdir(full):
                    shutil.rmtree(full)
                    deleted_audio_files.append(fn + os.sep)
                else:
                    os.remove(full)
                    deleted_audio_files.append(fn)

            report["planned_actions"].append({"type": "delete_duplicate_audio_files", "id": vid, "kept": keep_fn})

        report["planned_actions"].append({"type": "backup_dir", "path": backup_dir})
        report["deleted_audio_files"] = deleted_audio_files

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # Human-readable summary
    print("CSV duplicate ids:", len(csv_dups))
    print("AUDIO duplicate ids:", len(audio_dups))
    print("META inner-id mismatches:", len(meta_key_mismatch))
    if csv_dups:
        top = sorted(csv_dups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
        print("Top CSV duplicates:", [(k, len(v)) for k, v in top])
    if audio_dups:
        top = sorted(audio_dups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
        print("Top AUDIO duplicates:", [(k, len(v)) for k, v in top])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

