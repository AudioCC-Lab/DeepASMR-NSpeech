#!/usr/bin/env python3
"""Compute FD, FAD, KL, Inception Score, and CLAP on aligned manifests.

This is the cleaned public form of the test-set evaluator used for the paper.
Heavy evaluation dependencies are intentionally isolated from the annotation
environment; follow this directory's README before running it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def load_jsonl(path: Path, value_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        audio_id = str(row.get("audio_id") or "").strip()
        value = str(row.get(value_key) or "").strip()
        if not audio_id or not value:
            raise ValueError(f"invalid {path}:{line_number}")
        if audio_id in out:
            raise ValueError(f"duplicate audio_id {audio_id!r} in {path}")
        out[audio_id] = value
    return out


def align(
    reference: Dict[str, str],
    captions: Dict[str, str],
    generated: Dict[str, str],
    *,
    allow_missing: bool,
) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    ref_ids, caption_ids, generated_ids = set(reference), set(captions), set(generated)
    common = ref_ids & caption_ids & generated_ids
    if not allow_missing and not (ref_ids == caption_ids == generated_ids):
        raise ValueError(
            "manifest IDs differ: "
            f"reference={len(ref_ids)}, captions={len(caption_ids)}, "
            f"generated={len(generated_ids)}, intersection={len(common)}"
        )
    if not common:
        raise ValueError("manifests have no common audio IDs")
    ordered = sorted(common)
    return (
        {key: reference[key] for key in ordered},
        {key: captions[key] for key in ordered},
        {key: generated[key] for key in ordered},
    )


def numeric_mean(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        values = [float(item) for item in value.values()]
        return float(np.mean(values)) if values else None
    if isinstance(value, (list, tuple)):
        return [numeric_mean(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def distribution_metrics(
    reference: Dict[str, str],
    generated: Dict[str, str],
    *,
    sample_rate: int,
    backbone: str,
    num_workers: int,
    recalculate: bool,
) -> dict:
    try:
        import torch
        from audioldm_eval import EvaluationHelper
    except ImportError as exc:
        raise SystemExit(
            "global metric dependencies are unavailable; follow "
            "evaluation/model_benchmark/README.md"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = EvaluationHelper(sample_rate, device, backbone=backbone)
    raw = evaluator.main(
        generated,
        reference,
        recalculate=recalculate,
        num_workers=num_workers,
    )
    return {str(key): numeric_mean(value) for key, value in raw.items()}


def clap_metrics(
    captions: Dict[str, str],
    generated: Dict[str, str],
    *,
    checkpoint: str | None,
) -> dict:
    try:
        import laion_clap
        import librosa
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit(
            "CLAP dependencies are unavailable; follow "
            "evaluation/model_benchmark/README.md"
        ) from exc

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    if checkpoint:
        model.load_ckpt(ckpt=checkpoint, verbose=False)
    else:
        model.load_ckpt(model_id=1, verbose=False)
    model.eval()

    scores: Dict[str, float] = {}
    with torch.no_grad():
        for index, audio_id in enumerate(sorted(generated), start=1):
            waveform, _ = librosa.load(generated[audio_id], sr=48_000, mono=True)
            text_embedding = model.get_text_embedding([captions[audio_id]], use_tensor=False)
            audio_embedding = model.get_audio_embedding_from_data(x=[waveform], use_tensor=False)
            numerator = np.sum(audio_embedding * text_embedding, axis=1)
            denominator = np.linalg.norm(audio_embedding, axis=1) * np.linalg.norm(text_embedding, axis=1)
            scores[audio_id] = float((numerator / denominator)[0])
            if index % 100 == 0:
                print(f"CLAP [{index}/{len(generated)}]", flush=True)
    return {
        "CLAP_score": float(np.mean(list(scores.values()))),
        "per_audio": scores,
    }


def load_existing(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-captions", type=Path, required=True)
    parser.add_argument("--generated-audio", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "distribution", "clap"), default="all")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--backbone", choices=("cnn14", "mert"), default="cnn14")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clap-checkpoint", default=os.environ.get("CLAP_MODEL_PATH"))
    parser.add_argument("--recalculate", action="store_true")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Use only the ID intersection; not suitable for a comparable benchmark result",
    )
    args = parser.parse_args()

    reference = load_jsonl(args.reference_audio, "audio")
    captions = load_jsonl(args.reference_captions, "caption")
    generated = load_jsonl(args.generated_audio, "audio")
    reference, captions, generated = align(
        reference,
        captions,
        generated,
        allow_missing=args.allow_missing,
    )

    result = load_existing(args.out)
    result.update(
        {
            "model_id": args.model_id,
            "n_audio": len(generated),
            "sample_rate": args.sample_rate,
            "backbone": args.backbone,
            "strict_alignment": not args.allow_missing,
        }
    )
    if args.phase in ("all", "distribution"):
        result["distribution_metrics"] = distribution_metrics(
            reference,
            generated,
            sample_rate=args.sample_rate,
            backbone=args.backbone,
            num_workers=args.num_workers,
            recalculate=args.recalculate,
        )
    if args.phase in ("all", "clap"):
        result["clap"] = clap_metrics(
            captions,
            generated,
            checkpoint=args.clap_checkpoint,
        )

    summary = dict(result.get("distribution_metrics") or {})
    if result.get("clap"):
        summary["CLAP_score"] = result["clap"]["CLAP_score"]
    result["paper_metrics"] = {
        "FD": summary.get("frechet_distance"),
        "FAD": summary.get("frechet_audio_distance"),
        "KL": summary.get("kullback_leibler_divergence_softmax"),
        "ISc": summary.get("inception_score_mean"),
        "CLAP": summary.get("CLAP_score"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["paper_metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
