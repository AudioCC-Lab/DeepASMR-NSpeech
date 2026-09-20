#!/usr/bin/env python3
"""Build aligned reference/caption/generated-audio JSONL manifests.

The generated files contain local paths and belong in an untracked work
directory. They are inputs to ``global_metrics.py`` and are not dataset files.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_rows(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError(f"{path} must be a JSON list or contain a data list")


def load_models(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("models JSON must be a non-empty list or contain models")
    seen: set[str] = set()
    out: List[dict] = []
    for row in rows:
        model_id = str(row.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id):
            raise ValueError(f"invalid model id: {model_id!r}")
        if model_id in seen:
            raise ValueError(f"duplicate model id: {model_id}")
        seen.add(model_id)
        audio_dir_raw = str(row.get("audio_dir") or "").strip()
        if not audio_dir_raw:
            raise ValueError(f"model {model_id} is missing audio_dir")
        audio_dir = Path(audio_dir_raw).expanduser()
        out.append(
            {
                "id": model_id,
                "audio_dir": audio_dir,
                "filename": str(row.get("filename") or "{audio_id}.wav"),
            }
        )
    return out


def audio_id_for(row: dict) -> str:
    audio_id = str(row.get("audio_id") or "").strip()
    if audio_id:
        return audio_id
    wav = str(row.get("wav") or "").strip()
    if wav:
        return Path(wav).stem
    raise ValueError("label row has neither audio_id nor wav")


def resolve_reference(row: dict, dataset_root: Path) -> Path:
    wav = Path(str(row.get("wav") or "").strip()).expanduser()
    if not str(wav):
        raise ValueError(f"label row {audio_id_for(row)!r} has no wav")
    return wav if wav.is_absolute() else dataset_root / wav


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare(
    *,
    labels: Path,
    dataset_root: Path,
    models_json: Path,
    out_dir: Path,
    allow_missing: bool = False,
) -> dict:
    label_rows = load_rows(labels)
    models = load_models(models_json)

    index: Dict[str, dict] = {}
    for row in label_rows:
        audio_id = audio_id_for(row)
        if audio_id in index:
            raise ValueError(f"duplicate audio_id in labels: {audio_id}")
        caption = str(row.get("caption") or "").strip()
        if not caption:
            raise ValueError(f"missing caption for {audio_id}")
        index[audio_id] = {
            "reference": resolve_reference(row, dataset_root),
            "caption": caption,
        }

    missing_reference = [aid for aid, row in index.items() if not row["reference"].is_file()]
    model_paths: Dict[str, Dict[str, Path]] = {}
    missing_by_model: Dict[str, List[str]] = {}
    for model in models:
        paths = {
            aid: model["audio_dir"] / model["filename"].format(audio_id=aid)
            for aid in index
        }
        model_paths[model["id"]] = paths
        missing_by_model[model["id"]] = [aid for aid, path in paths.items() if not path.is_file()]

    missing_total = len(missing_reference) + sum(len(v) for v in missing_by_model.values())
    if missing_total and not allow_missing:
        details = {
            "missing_reference": missing_reference[:20],
            "missing_by_model": {k: v[:20] for k, v in missing_by_model.items() if v},
        }
        raise FileNotFoundError(
            "strict alignment failed; pass --allow-missing only for diagnostics: "
            + json.dumps(details, ensure_ascii=False)
        )

    usable = [aid for aid in index if index[aid]["reference"].is_file()]
    if allow_missing:
        usable = [
            aid for aid in usable
            if all(model_paths[model["id"]][aid].is_file() for model in models)
        ]
    if not usable:
        raise ValueError("no fully aligned audio IDs remain")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        out_dir / "reference_audio.jsonl",
        ({"audio_id": aid, "audio": str(index[aid]["reference"].resolve())} for aid in usable),
    )
    write_jsonl(
        out_dir / "reference_captions.jsonl",
        ({"audio_id": aid, "caption": index[aid]["caption"]} for aid in usable),
    )
    for model in models:
        write_jsonl(
            out_dir / f"{model['id']}.jsonl",
            (
                {"audio_id": aid, "audio": str(model_paths[model["id"]][aid].resolve())}
                for aid in usable
            ),
        )

    report: Dict[str, Any] = {
        "labels": str(labels),
        "dataset_root": str(dataset_root),
        "n_labels": len(index),
        "n_aligned": len(usable),
        "strict": not allow_missing,
        "missing_reference": len(missing_reference),
        "missing_by_model": {k: len(v) for k, v in missing_by_model.items()},
        "models": [model["id"] for model in models],
    }
    (out_dir / "alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Released test_label.json")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True, help="Model directory registry JSON")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Evaluate only the intersection; do not use for a comparable benchmark result",
    )
    args = parser.parse_args()
    report = prepare(
        labels=args.labels,
        dataset_root=args.dataset_root,
        models_json=args.models,
        out_dir=args.out_dir,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
