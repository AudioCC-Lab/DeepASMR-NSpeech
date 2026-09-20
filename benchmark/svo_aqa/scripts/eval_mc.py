#!/usr/bin/env python3
"""Score one or more SVO-AQA prediction files against bank.json."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

TASKS = ("verb_mc", "subject_material_mc", "object_material_mc")


def prediction_rows(raw: Any) -> Iterable[dict]:
    if isinstance(raw, dict) and isinstance(raw.get("results"), list):
        return raw["results"]
    if isinstance(raw, list):
        return raw
    return []


def load_predictions(path: Path) -> Dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    if isinstance(raw, dict) and not isinstance(raw.get("results"), list):
        for key, value in raw.items():
            if isinstance(value, str):
                out[str(key)] = value.strip().upper()
            elif isinstance(value, dict):
                pred = value.get("pred_logprob") or value.get("pred_label") or value.get("pred")
                if pred:
                    out[str(key)] = str(pred).strip().upper()
        return out
    for row in prediction_rows(raw):
        question_id = row.get("question_id")
        pred = row.get("pred_logprob") or row.get("pred_label") or row.get("pred")
        if question_id and pred:
            out[str(question_id)] = str(pred).strip().upper()
    return out


def load_subset_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("questions") or payload.get("question_ids") or []
    else:
        raise ValueError(f"invalid subset JSON: {path}")
    return {
        str(row.get("question_id") if isinstance(row, dict) else row)
        for row in rows
    }


def option_class(question: dict, label: str | None) -> str | None:
    for option in question.get("options") or []:
        if str(option.get("label") or "").upper() == label:
            value = option.get("verb_class")
            return str(value) if value else None
    return None


def score_run(questions: list[dict], predictions: Dict[str, str]) -> dict:
    correct = 0
    task_total: Counter = Counter()
    task_hits: Counter = Counter()
    superclass_total = superclass_hits = 0

    for question in questions:
        question_id = str(question["question_id"])
        task = str(question.get("task") or "unknown")
        pred = predictions.get(question_id)
        gold = str(question.get("gold_label") or "").upper()
        task_total[task] += 1
        if pred == gold:
            correct += 1
            task_hits[task] += 1
        if task == "verb_mc":
            gold_class = option_class(question, gold)
            pred_class = option_class(question, pred)
            if gold_class:
                superclass_total += 1
                superclass_hits += int(pred_class == gold_class)

    n_questions = len(questions)
    per_task = {
        task: {
            "n": task_total[task],
            "accuracy": task_hits[task] / task_total[task] if task_total[task] else 0.0,
        }
        for task in sorted(task_total)
    }
    macro_values = [per_task[task]["accuracy"] for task in TASKS if per_task.get(task, {}).get("n")]
    return {
        "n_questions": n_questions,
        "n_scored": sum(question["question_id"] in predictions for question in questions),
        "accuracy": correct / n_questions if n_questions else 0.0,
        "macro_accuracy": sum(macro_values) / len(macro_values) if macro_values else 0.0,
        "verb_superclass": {
            "n": superclass_total,
            "accuracy": superclass_hits / superclass_total if superclass_total else 0.0,
        },
        "per_task": per_task,
    }


def average_runs(runs: list[dict]) -> dict:
    first = runs[0]
    per_task = {}
    for task in first["per_task"]:
        values = [run["per_task"][task]["accuracy"] for run in runs]
        per_task[task] = {
            "n": first["per_task"][task]["n"],
            "accuracy": statistics.fmean(values),
            "std": statistics.pstdev(values),
        }
    superclass_values = [run["verb_superclass"]["accuracy"] for run in runs]
    accuracy_values = [run["accuracy"] for run in runs]
    macro_values = [run["macro_accuracy"] for run in runs]
    return {
        "n_questions": first["n_questions"],
        "n_runs": len(runs),
        "n_scored": [run["n_scored"] for run in runs],
        "accuracy": statistics.fmean(accuracy_values),
        "accuracy_std": statistics.pstdev(accuracy_values),
        "macro_accuracy": statistics.fmean(macro_values),
        "macro_accuracy_std": statistics.pstdev(macro_values),
        "verb_superclass": {
            "n": first["verb_superclass"]["n"],
            "accuracy": statistics.fmean(superclass_values),
            "std": statistics.pstdev(superclass_values),
        },
        "per_task": per_task,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True, help="bank.json path")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Optional summary JSON")
    args = parser.parse_args()

    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    questions = list(bank["questions"])
    subset_ids = load_subset_ids(args.subset)
    if subset_ids is not None:
        available = {str(question["question_id"]) for question in questions}
        missing = subset_ids - available
        if missing:
            raise SystemExit(f"subset contains {len(missing)} IDs absent from bank")
        questions = [question for question in questions if str(question["question_id"]) in subset_ids]

    run_scores = [score_run(questions, load_predictions(path)) for path in args.predictions]
    summary = run_scores[0] if len(run_scores) == 1 else average_runs(run_scores)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
