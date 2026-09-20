#!/usr/bin/env python3
"""Measure agreement between two independent manual-GT annotation files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.annotation_quality.common import (  # noqa: E402
    build_identifier_index,
    canonical_verb,
    cohen_kappa,
    default_lexicons,
    extract_svo,
    is_completed_gt,
    load_alias_groups,
    load_rows,
    match_row,
    normalize_text,
    noun_match_stage,
    preferred_identifier,
    supported_accuracy,
    write_json,
)


def calculate(args: argparse.Namespace) -> Dict[str, Any]:
    left_rows = [row for row in load_rows(args.annotator_a) if is_completed_gt(row)]
    right_rows = [row for row in load_rows(args.annotator_b) if is_completed_gt(row)]
    right_index = build_identifier_index(right_rows)
    lexicon, verb_aliases = default_lexicons(REPO_ROOT)
    noun_aliases = load_alias_groups(args.noun_aliases)

    verb_left: List[Optional[str]] = []
    verb_right: List[Optional[str]] = []
    class_left: List[Optional[str]] = []
    class_right: List[Optional[str]] = []
    entity_matches = {"subject": {"strict": 0, "alias": 0}, "object": {"strict": 0, "alias": 0}}
    matched_ids: List[str] = []

    for left_row in left_rows:
        right_row = match_row(left_row, right_index)
        if right_row is None:
            continue
        matched_ids.append(preferred_identifier(left_row))
        left_svo = extract_svo(left_row, ground_truth=True)
        right_svo = extract_svo(right_row, ground_truth=True)
        left_verb = canonical_verb(left_svo["verb"], lexicon, verb_aliases)
        right_verb = canonical_verb(right_svo["verb"], lexicon, verb_aliases)
        verb_left.append(left_verb)
        verb_right.append(right_verb)
        class_left.append(lexicon.class_of(left_verb) if left_verb else None)
        class_right.append(lexicon.class_of(right_verb) if right_verb else None)
        for field in ("subject", "object"):
            stage = noun_match_stage(left_svo[field], right_svo[field], noun_aliases)
            entity_matches[field]["strict"] += int(stage == "strict")
            entity_matches[field]["alias"] += int(stage in {"strict", "alias"})

    total = len(matched_ids)
    if total == 0:
        raise ValueError("No completed rows could be aligned between annotators")
    entities: Dict[str, Any] = {}
    for field, counts in entity_matches.items():
        entities[field] = {
            "n": total,
            "strict_agreement": round(counts["strict"] / total, 6),
            "alias_normalized_agreement": round(counts["alias"] / total, 6),
        }
    return {
        "dataset": "DeepASMR-NSpeech",
        "evaluation": "inter-annotator agreement on independently produced manual GT",
        "counts": {
            "annotator_a_completed": len(left_rows),
            "annotator_b_completed": len(right_rows),
            "aligned": total,
        },
        "verb": {
            "exact_agreement": round(supported_accuracy(verb_left, verb_right), 6),
            "cohen_kappa": round(cohen_kappa(verb_left, verb_right), 6),
        },
        "superclass": {
            "exact_agreement": round(supported_accuracy(class_left, class_right), 6),
            "cohen_kappa": round(cohen_kappa(class_left, class_right), 6),
        },
        "subject": entities["subject"],
        "object": entities["object"],
        "notes": [
            "Open-set noun agreement is reported as pairwise agreement, not Cohen's kappa.",
            "If only one annotator labeled the data, do not report this as inter-annotator agreement.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--noun-aliases",
        type=Path,
        default=Path(__file__).with_name("data") / "noun_alias.v1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = calculate(args)
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
