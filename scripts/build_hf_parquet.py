#!/usr/bin/env python3
"""Build Hugging Face Parquet shards with embedded DeepASMR-NSpeech audio.

The output follows the public DeepASMR release pattern: audio bytes and
annotations are stored together in sharded Parquet files that can be loaded
directly with ``datasets.load_dataset``.

Run ``prepare_deepasmr_nspeech_hf.py --labels-only`` first to create the
dataset card, public JSON labels, SVO-AQA files, and vocabulary reference. This
script then adds ``parquet/train-*.parquet`` and ``parquet/test-*.parquet``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_rows(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list or {{'data': [...]}} in {path}")
    return rows


def build_creator_map(*row_groups: Sequence[dict]) -> Dict[str, str]:
    authors = sorted(
        {
            str(row.get("author") or "").strip()
            for rows in row_groups
            for row in rows
            if str(row.get("author") or "").strip()
        }
    )
    return {author: f"creator_{index:02d}" for index, author in enumerate(authors, 1)}


def derive_audio_id(wav: Path) -> str:
    """Return a stable public ID without machine or creator directory names."""
    parts = list(wav.parts)
    if "splitted" in parts:
        tail = parts[parts.index("splitted") + 1 :]
        tail[-1] = Path(tail[-1]).stem
        candidate = "__".join(tail)
    else:
        candidate = wav.stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("_")


def public_row(row: dict, creator_map: Dict[str, str]) -> dict:
    wav = Path(str(row.get("wav") or "").strip())
    if not wav.is_file():
        raise FileNotFoundError(wav)
    audio_id = str(row.get("audio_id") or derive_audio_id(wav)).strip()
    if not audio_id:
        raise ValueError(f"Unable to derive audio_id from {wav}")
    return {
        "audio_id": audio_id,
        "audio": str(wav),
        "caption": str(row.get("caption") or "").strip(),
        "verb": str(row.get("verb") or "").strip(),
        "subject": str(row.get("subject") or "").strip(),
        "object": str(row.get("object") or "").strip(),
        "creator_id": creator_map.get(str(row.get("author") or "").strip(), "creator_unknown"),
    }


def validate_audio_ids(rows_by_split: Dict[str, Sequence[dict]]) -> None:
    split_ids: Dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        counts = Counter(str(row["audio_id"]) for row in rows)
        duplicates = sorted(audio_id for audio_id, count in counts.items() if count > 1)
        if duplicates:
            preview = ", ".join(duplicates[:5])
            raise ValueError(f"{split} contains {len(duplicates)} duplicate audio_id values: {preview}")
        split_ids[split] = set(counts)
    if "train" in split_ids and "test" in split_ids:
        overlap = sorted(split_ids["train"] & split_ids["test"])
        if overlap:
            preview = ", ".join(overlap[:5])
            raise ValueError(f"train/test audio_id overlap ({len(overlap)}): {preview}")


def shard_rows(rows: Sequence[dict], target_bytes: int) -> List[List[dict]]:
    shards: List[List[dict]] = []
    current: List[dict] = []
    current_bytes = 0
    for row in rows:
        size = Path(row["audio"]).stat().st_size
        if current and current_bytes + size > target_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(row)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def require_parquet_dependencies():
    try:
        import pyarrow.parquet as pq
        from datasets import Audio, Dataset, Features, Value
        from datasets.table import embed_table_storage
    except ImportError as exc:
        raise SystemExit(
            "Install the Hugging Face release dependencies first: "
            "python -m pip install 'datasets>=3.5,<4' 'pyarrow>=21,<22'"
        ) from exc
    return pq, Audio, Dataset, Features, Value, embed_table_storage


def parquet_row_count(path: Path, pq: Any) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def write_split(
    split: str,
    rows: Sequence[dict],
    out_dir: Path,
    *,
    target_bytes: int,
    compression: str,
    resume: bool,
) -> dict:
    pq, Audio, Dataset, Features, Value, embed_table_storage = require_parquet_dependencies()
    features = Features(
        {
            "audio_id": Value("string"),
            "audio": Audio(decode=False),
            "caption": Value("string"),
            "verb": Value("string"),
            "subject": Value("string"),
            "object": Value("string"),
            "creator_id": Value("string"),
        }
    )
    shards = shard_rows(rows, target_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)
    written_bytes = 0
    for index, shard in enumerate(shards):
        out = out_dir / f"{split}-{index:05d}-of-{len(shards):05d}.parquet"
        if resume and out.is_file() and parquet_row_count(out, pq) == len(shard):
            written_bytes += out.stat().st_size
            print(f"[SKIP] {out.name}: {len(shard)} rows", flush=True)
            continue
        tmp = out.with_suffix(out.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        dataset = Dataset.from_list(list(shard), features=features)
        embedded = embed_table_storage(dataset.data.table)
        pq.write_table(embedded, tmp, compression=compression)
        if parquet_row_count(tmp, pq) != len(shard):
            raise RuntimeError(f"Row-count validation failed for {tmp}")
        os.replace(tmp, out)
        written_bytes += out.stat().st_size
        print(
            f"[WRITE] {out.name}: {len(shard)} rows, "
            f"{out.stat().st_size / 2**30:.2f} GiB",
            flush=True,
        )
    return {
        "rows": len(rows),
        "shards": len(shards),
        "parquet_bytes": written_bytes,
        "parquet_gib": round(written_bytes / 2**30, 3),
    }


def write_metadata_jsonl(split: str, rows: Iterable[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            public = {key: value for key, value in row.items() if key != "audio"}
            public["audio"] = Path(row["audio"]).name
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-label", type=Path, required=True)
    parser.add_argument("--test-label", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument(
        "--target-shard-size-gib",
        type=float,
        default=1.0,
        help="Approximate uncompressed audio bytes per shard (default: 1 GiB).",
    )
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.target_shard_size_gib) or args.target_shard_size_gib <= 0:
        raise SystemExit("--target-shard-size-gib must be a positive finite number")
    train_raw = load_rows(args.train_label)
    test_raw = load_rows(args.test_label)
    creator_map = build_creator_map(train_raw, test_raw)
    rows_by_split = {
        "train": [public_row(row, creator_map) for row in train_raw],
        "test": [public_row(row, creator_map) for row in test_raw],
    }
    validate_audio_ids(rows_by_split)
    selected = ("train", "test") if args.split == "all" else (args.split,)
    target_bytes = int(args.target_shard_size_gib * 2**30)
    stats: Dict[str, Any] = {
        "format": "embedded-audio-parquet",
        "compression": args.compression,
        "target_shard_size_gib": args.target_shard_size_gib,
        "creator_count": len(creator_map),
        "splits": {},
    }
    for split in selected:
        rows = rows_by_split[split]
        write_metadata_jsonl(split, rows, args.out_root / "metadata")
        stats["splits"][split] = write_split(
            split,
            rows,
            args.out_root / "parquet",
            target_bytes=target_bytes,
            compression=args.compression,
            resume=args.resume,
        )
    manifest_path = args.out_root / "parquet_manifest.json"
    manifest_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote {manifest_path}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
