#!/usr/bin/env python3
"""Compare pipeline SVO annotations with independently hand-labeled GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation.common.info_common import (  # noqa: E402
    DEFAULT_SILICONFLOW_BASE_URL,
    TokenUsageAccumulator,
    call_llm_json_with_usage,
    fill_template,
    make_openai_client,
)
from annotation.common.prompt_util import load_marked_prompt  # noqa: E402
from evaluation.annotation_quality.common import (  # noqa: E402
    build_identifier_index,
    canonical_verb,
    default_lexicons,
    entity_summary,
    extract_svo,
    is_completed_gt,
    load_alias_groups,
    load_rows,
    match_row,
    metric_pair,
    normalize_text,
    noun_match_stage,
    preferred_identifier,
    write_json,
)


class NounAdjudicator:
    """Optional cached LLM judge for noun pairs unresolved by public rules."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.enabled = bool(args.use_llm)
        self.cache_path: Optional[Path] = args.llm_cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and self.cache_path.is_file():
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                self.cache = {str(key): dict(item) for key, item in value.items() if isinstance(item, dict)}
        self.system_prompt, self.user_prompt = load_marked_prompt(args.llm_prompt)
        self.prompt_sha256 = hashlib.sha256(args.llm_prompt.read_bytes()).hexdigest()
        self.model = args.model
        self.base_url = args.base_url
        self.usage = TokenUsageAccumulator()
        self.client: Any = None
        self.errors = 0
        if self.enabled:
            api_key = os.environ.get(args.api_key_env, "").strip()
            if not api_key:
                raise ValueError(f"--use-llm requires the API key in environment variable {args.api_key_env}")
            if not self.model:
                raise ValueError("--use-llm requires --model (or ANNOTATION_QA_MODEL)")
            self.client = make_openai_client(api_key, self.base_url)

    @staticmethod
    def _key(left: Any, right: Any) -> str:
        return json.dumps([normalize_text(left), normalize_text(right)], ensure_ascii=False)

    def judge(self, left: Any, right: Any) -> Tuple[bool, str]:
        key = self._key(left, right)
        cached = self.cache.get(key)
        if cached is not None:
            value = cached.get("equivalent")
            equivalent = value if isinstance(value, bool) else str(value).strip().lower() == "true"
            return equivalent, "llm_cache"
        if not self.enabled:
            return False, "unresolved"
        values = {"GT_NOUN": str(left), "PRED_NOUN": str(right)}
        user_text = self.user_prompt
        for name, value in values.items():
            user_text = user_text.replace("{{" + name + "}}", value)
        user_text = fill_template(user_text, values)
        try:
            result = call_llm_json_with_usage(
                client=self.client,
                model=self.model,
                system_prompt=self.system_prompt,
                user_text=user_text,
                usage_acc=self.usage,
                stage="noun_equivalence",
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=256,
            )
            value = result.get("equivalent")
            equivalent = value if isinstance(value, bool) else str(value).strip().lower() == "true"
            self.cache[key] = {
                "equivalent": equivalent,
                "reason": str(result.get("reason") or ""),
                "model": self.model,
                "prompt_file": self.prompt_path_name,
            }
            return equivalent, "llm"
        except Exception as exc:  # keep one API failure from invalidating the full audit
            self.errors += 1
            self.cache[key] = {"equivalent": False, "error_type": type(exc).__name__, "model": self.model}
            return False, "llm_error"

    @property
    def prompt_path_name(self) -> str:
        return "noun_equivalence.txt"

    def save(self) -> None:
        if self.cache_path:
            write_json(self.cache_path, self.cache)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    predictions = load_rows(args.predictions)
    ground_truth = load_rows(args.ground_truth)
    completed = [row for row in ground_truth if is_completed_gt(row)]
    incomplete = len(ground_truth) - len(completed)
    if incomplete and not args.allow_incomplete_gt:
        raise ValueError(
            f"Ground truth contains {incomplete} incomplete/pending rows. Finish them or use --allow-incomplete-gt."
        )
    if not completed:
        raise ValueError("No completed human-GT rows were found")

    prediction_index = build_identifier_index(predictions)
    lexicon, verb_aliases = default_lexicons(REPO_ROOT)
    noun_aliases = load_alias_groups(args.noun_aliases)
    adjudicator = NounAdjudicator(args)

    verb_strict_gold: List[Optional[str]] = []
    verb_strict_pred: List[Optional[str]] = []
    verb_gold: List[Optional[str]] = []
    verb_pred: List[Optional[str]] = []
    class_gold: List[Optional[str]] = []
    class_pred: List[Optional[str]] = []
    entity_stages: Dict[str, List[Optional[str]]] = {"subject": [], "object": []}
    entity_llm_matches = Counter()
    full_strict = 0
    full_resolved = 0
    missing_predictions: List[str] = []

    for gt_row in completed:
        pred_row = match_row(gt_row, prediction_index)
        gt_svo = extract_svo(gt_row, ground_truth=True)
        pred_svo = extract_svo(pred_row or {}, ground_truth=False)
        if pred_row is None:
            missing_predictions.append(preferred_identifier(gt_row))

        gt_verb_raw = normalize_text(gt_svo["verb"])
        pred_verb_raw = normalize_text(pred_svo["verb"])
        gt_verb = canonical_verb(gt_svo["verb"], lexicon, verb_aliases)
        pred_verb = canonical_verb(pred_svo["verb"], lexicon, verb_aliases)
        if gt_verb is None:
            raise ValueError(
                f"Manual GT verb {gt_svo['verb']!r} for {preferred_identifier(gt_row)!r} "
                "is outside the released closed vocabulary"
            )
        verb_strict_gold.append(gt_verb_raw)
        verb_strict_pred.append(pred_verb_raw)
        verb_gold.append(gt_verb)
        verb_pred.append(pred_verb)
        class_gold.append(lexicon.class_of(gt_verb) if gt_verb else None)
        class_pred.append(lexicon.class_of(pred_verb) if pred_verb else None)

        row_strict = gt_verb_raw == pred_verb_raw
        row_resolved = gt_verb is not None and gt_verb == pred_verb
        for field in ("subject", "object"):
            if pred_row is None:
                entity_stages[field].append(None)
                row_strict = False
                row_resolved = False
                continue
            stage = noun_match_stage(gt_svo[field], pred_svo[field], noun_aliases)
            entity_stages[field].append(stage)
            if stage is None:
                equivalent, resolution = adjudicator.judge(gt_svo[field], pred_svo[field])
                if equivalent:
                    entity_llm_matches[field] += 1
                    row_resolved = row_resolved and True
                else:
                    row_resolved = False
                if resolution in {"llm", "llm_cache"} and equivalent:
                    entity_stages[field][-1] = resolution
            else:
                row_resolved = row_resolved and True
            row_strict = row_strict and stage == "strict"
        full_strict += int(row_strict)
        full_resolved += int(row_resolved)

    adjudicator.save()
    subject = entity_summary(entity_stages["subject"], llm_matches=entity_llm_matches["subject"])
    object_metrics = entity_summary(entity_stages["object"], llm_matches=entity_llm_matches["object"])
    for field, metrics in (("subject", subject), ("object", object_metrics)):
        metrics["llm_correct"] = entity_llm_matches[field]

    total = len(completed)
    return {
        "dataset": "DeepASMR-NSpeech",
        "evaluation": "annotation quality against independent human ground truth",
        "ground_truth_requirement": "GT must be manually annotated; pipeline-generated labels are not valid GT",
        "counts": {
            "gt_rows_total": len(ground_truth),
            "gt_rows_completed": total,
            "gt_rows_incomplete_excluded": incomplete,
            "prediction_rows": len(predictions),
            "missing_predictions": len(missing_predictions),
            "missing_prediction_item_ids": missing_predictions[:100],
        },
        "verb": {
            "strict": metric_pair(verb_strict_gold, verb_strict_pred),
            "alias_normalized_fine": metric_pair(verb_gold, verb_pred),
            "superclass": metric_pair(class_gold, class_pred),
            "unrecognized_gt_verbs": sum(label is None for label in verb_gold),
            "unrecognized_predicted_verbs": sum(label is None for label in verb_pred),
        },
        "subject": subject,
        "object": object_metrics,
        "full_svo": {
            "n": total,
            "strict_accuracy": round(full_strict / total, 6),
            "alias_or_llm_accuracy": round(full_resolved / total, 6),
        },
        "noun_matching": {
            "alias_file": args.noun_aliases.name if args.noun_aliases else None,
            "alias_sha256": (
                hashlib.sha256(args.noun_aliases.read_bytes()).hexdigest()
                if args.noun_aliases
                else None
            ),
            "alias_entries": len(noun_aliases),
            "llm_enabled_for_uncached_pairs": adjudicator.enabled,
            "llm_model": adjudicator.model if adjudicator.enabled else None,
            "llm_prompt_sha256": adjudicator.prompt_sha256,
            "llm_temperature": 0.0,
            "llm_max_tokens": 256,
            "llm_errors": adjudicator.errors,
            "llm_usage": adjudicator.usage.to_dict(),
        },
        "notes": [
            "Material descriptions are intentionally excluded from this annotation-quality audit.",
            "Report strict and alias/LLM-assisted entity scores separately.",
            "Operational success/failure summaries do not measure annotation correctness.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Pipeline-produced labels")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Completed manual-GT JSON/JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--noun-aliases",
        type=Path,
        default=Path(__file__).with_name("data") / "noun_alias.v1.csv",
        help="Frozen group_id,alias CSV used for the v1 paper audit",
    )
    parser.add_argument("--allow-incomplete-gt", action="store_true")
    parser.add_argument("--llm-cache", type=Path, default=None, help="Local adjudication cache; do not publish if it reveals private GT")
    parser.add_argument("--use-llm", action="store_true", help="Call an LLM only for unresolved noun pairs")
    parser.add_argument(
        "--llm-prompt",
        type=Path,
        default=Path(__file__).with_name("prompts") / "noun_equivalence.txt",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ANNOTATION_QA_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL))
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY", help="Name of environment variable containing the key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
