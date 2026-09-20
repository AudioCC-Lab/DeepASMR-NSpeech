#!/usr/bin/env python3
"""Create a blind, verb-stratified manual-GT annotation template."""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.annotation_quality.common import (  # noqa: E402
    canonical_verb,
    default_lexicons,
    extract_svo,
    load_rows,
    preferred_identifier,
    write_json,
)


def stratified_sample(
    rows: Sequence[Mapping[str, Any]], *, sample_size: int, seed: int
) -> Tuple[List[Mapping[str, Any]], Dict[str, int]]:
    """Round-robin over shuffled verb buckets for near-equal representation."""
    lexicon, aliases = default_lexicons(REPO_ROOT)
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    seen_ids = set()
    for row in rows:
        item_id = preferred_identifier(row)
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        verb = canonical_verb(extract_svo(row, ground_truth=False).get("verb"), lexicon, aliases)
        if verb is not None:
            buckets[verb].append(row)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    verbs = sorted(buckets)
    rng.shuffle(verbs)
    target = min(sample_size, sum(len(bucket) for bucket in buckets.values()))
    selected: List[Mapping[str, Any]] = []
    positions = {verb: 0 for verb in verbs}
    while len(selected) < target:
        progress = False
        for verb in verbs:
            position = positions[verb]
            if position >= len(buckets[verb]):
                continue
            selected.append(buckets[verb][position])
            positions[verb] += 1
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break
    rng.shuffle(selected)
    counts = Counter(
        canonical_verb(extract_svo(row, ground_truth=False).get("verb"), lexicon, aliases)
        for row in selected
    )
    return selected, {str(key): counts[key] for key in sorted(counts) if key is not None}


def build_template(
    rows: Sequence[Mapping[str, Any]], *, split: str, include_context: bool
) -> List[Dict[str, Any]]:
    """Hide pipeline SVO labels by default to reduce anchoring bias."""
    output: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {
            "item_id": preferred_identifier(row),
            "wav": row.get("wav") or row.get("audio") or row.get("audio_path"),
            "split": split,
            "gt": {"subject": None, "verb": None, "object": None},
            "annotation": {
                "annotator_id": None,
                "status": "pending",
                "notes": None,
            },
        }
        if row.get("keyframe") or row.get("image"):
            item["keyframe"] = row.get("keyframe") or row.get("image")
        if include_context:
            item["context"] = {
                "segment_label": row.get("seg_label") or row.get("segment_label"),
                "title": row.get("title"),
                "visual_description": row.get("visual_description"),
            }
        output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Pipeline label JSON/JSONL used only for sampling strata")
    parser.add_argument("--out", type=Path, required=True, help="Blank manual-GT template")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=206)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Expose non-SVO context. Keep disabled for the least biased audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    rows = load_rows(args.predictions)
    selected, counts = stratified_sample(rows, sample_size=args.sample_size, seed=args.seed)
    if not selected:
        raise SystemExit("No rows with a recognized closed-set verb were available for sampling")
    payload = {
        "meta": {
            "dataset": "DeepASMR-NSpeech",
            "purpose": "manual annotation-quality ground truth",
            "gt_source": "independent human annotation required",
            "split": args.split,
            "seed": args.seed,
            "requested_sample_size": args.sample_size,
            "selected_sample_size": len(selected),
            "sampling": "near-equal verb-stratified sampling without replacement",
            "pipeline_svo_hidden": True,
            "selected_stratum_counts": counts,
            "instructions": (
                "Listen to the clip and inspect only the evidence allowed by the protocol. "
                "Fill gt.subject, gt.verb, and gt.object; choose gt.verb from the released "
                "closed vocabulary, and object may be null. Then set "
                "annotation.status to complete. Do not copy pipeline predictions."
            ),
        },
        "data": build_template(selected, split=args.split, include_context=args.include_context),
    }
    write_json(args.out, payload)
    print(f"Wrote {len(selected)} blank human-GT rows to {args.out}")


if __name__ == "__main__":
    main()
