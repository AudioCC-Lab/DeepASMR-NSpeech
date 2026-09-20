"""Shared, dependency-light helpers for the annotation-quality audit."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from annotation.common.info_common import (
    VerbClassLexicon,
    load_verb_alias_map,
    load_verb_class_lexicon,
    norm_token,
    resolve_verb_canonical,
)


NULL_LABEL = "<null>"
MISSING_LABEL = "<missing>"


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON/JSONL list, or the list stored under data/events/items."""
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(dict(value))
        return rows

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("data", "events", "items", "annotations"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
    raise ValueError(f"Expected a JSON list or an object containing data/events/items: {path}")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> Optional[str]:
    """Normalize presentation differences without performing semantic matching."""
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text or None


def identifiers(row: Mapping[str, Any]) -> List[str]:
    """Return stable candidate identifiers from strongest to weakest."""
    values: List[str] = []
    for key in ("item_id", "audio_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            values.append(str(value).strip())
    wav = row.get("wav") or row.get("audio") or row.get("audio_path")
    if wav is not None and str(wav).strip():
        normalized = str(PurePosixPath(str(wav).replace("\\", "/")))
        values.extend((normalized, PurePosixPath(normalized).stem))
    return list(dict.fromkeys(values))


def preferred_identifier(row: Mapping[str, Any]) -> str:
    candidates = identifiers(row)
    if not candidates:
        raise ValueError("Every audited row must have item_id, audio_id, id, or wav")
    return candidates[0]


def build_identifier_index(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[Mapping[str, Any]]]:
    """Map every identifier to a row; ambiguous identifiers map to None."""
    index: Dict[str, Optional[Mapping[str, Any]]] = {}
    for row in rows:
        for key in identifiers(row):
            if key in index and index[key] is not row:
                index[key] = None
            else:
                index[key] = row
    return index


def match_row(row: Mapping[str, Any], index: Mapping[str, Optional[Mapping[str, Any]]]) -> Optional[Mapping[str, Any]]:
    for key in identifiers(row):
        match = index.get(key)
        if match is not None:
            return match
    return None


def extract_svo(row: Mapping[str, Any], *, ground_truth: bool) -> Dict[str, Any]:
    nested_keys = ("gt", "ground_truth") if ground_truth else ("prediction", "pred", "pipeline")
    source: Mapping[str, Any] = row
    for key in nested_keys:
        value = row.get(key)
        if isinstance(value, dict):
            source = value
            break
    return {field: source.get(field) for field in ("subject", "verb", "object")}


def is_completed_gt(row: Mapping[str, Any]) -> bool:
    annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
    status = normalize_text(annotation.get("status") or row.get("status"))
    if status in {"pending", "skip", "skipped", "invalid", "excluded"}:
        return False
    svo = extract_svo(row, ground_truth=True)
    return normalize_text(svo["subject"]) is not None and normalize_text(svo["verb"]) is not None


def load_alias_groups(path: Optional[Path]) -> Dict[str, str]:
    """Load group_id,alias or canonical,alias rows for open-set noun matching."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    aliases: Dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            group = normalize_text(row.get("group_id") or row.get("canonical"))
            alias = normalize_text(row.get("alias"))
            if group and alias:
                previous = aliases.get(alias)
                if previous is not None and previous != group:
                    raise ValueError(f"Alias {alias!r} appears in multiple groups in {path}")
                aliases[alias] = group
    return aliases


def noun_match_stage(gt: Any, pred: Any, aliases: Mapping[str, str]) -> Optional[str]:
    """Return strict/alias for a match, otherwise None."""
    left = normalize_text(gt)
    right = normalize_text(pred)
    if left == right:
        return "strict"
    if left is None or right is None:
        return None
    left_group = aliases.get(left)
    right_group = aliases.get(right)
    if left_group and right_group and left_group == right_group:
        return "alias"
    return None


def canonical_verb(raw: Any, lexicon: VerbClassLexicon, aliases: Mapping[str, str]) -> Optional[str]:
    if raw is None:
        return None
    return resolve_verb_canonical(str(raw), lexicon, dict(aliases))


def supported_accuracy(gold: Sequence[Optional[str]], pred: Sequence[Optional[str]]) -> float:
    if len(gold) != len(pred):
        raise ValueError("gold and pred lengths differ")
    if not gold:
        return 0.0
    return sum(left == right for left, right in zip(gold, pred)) / len(gold)


def macro_f1(gold: Sequence[Optional[str]], pred: Sequence[Optional[str]]) -> float:
    """Macro-F1 over labels with ground-truth support; missing predictions are wrong."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred lengths differ")
    labels = sorted({label for label in gold if label is not None})
    if not labels:
        return 0.0
    values: List[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values)


def metric_pair(gold: Sequence[Optional[str]], pred: Sequence[Optional[str]]) -> Dict[str, Any]:
    per_label: Dict[str, Dict[str, Any]] = {}
    for label in sorted({item for item in gold if item is not None}):
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(g == label for g in gold),
        }
    return {
        "n": len(gold),
        "accuracy": round(supported_accuracy(gold, pred), 6),
        "macro_f1": round(macro_f1(gold, pred), 6),
        "per_label": per_label,
    }


def entity_summary(stages: Sequence[Optional[str]], *, llm_matches: int = 0) -> Dict[str, Any]:
    total = len(stages)
    strict = sum(stage == "strict" for stage in stages)
    alias = sum(stage in {"strict", "alias"} for stage in stages)
    resolved = alias + llm_matches
    return {
        "n": total,
        "strict_correct": strict,
        "strict_accuracy": round(strict / total, 6) if total else 0.0,
        "alias_correct": alias,
        "alias_accuracy": round(alias / total, 6) if total else 0.0,
        "alias_or_llm_correct": resolved,
        "alias_or_llm_accuracy": round(resolved / total, 6) if total else 0.0,
        "match_stage_counts": dict(sorted(Counter(stage or "unresolved" for stage in stages).items())),
    }


def cohen_kappa(left: Sequence[Optional[str]], right: Sequence[Optional[str]]) -> float:
    if len(left) != len(right):
        raise ValueError("annotation lengths differ")
    if not left:
        return 0.0
    observed = supported_accuracy(left, right)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(left_counts[label] * right_counts[label] for label in labels) / (len(left) ** 2)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def default_lexicons(repo_root: Path) -> Tuple[VerbClassLexicon, Dict[str, str]]:
    lexicon = load_verb_class_lexicon(repo_root / "vocab" / "asmr_verb_classes_en.csv")
    aliases = load_verb_alias_map(repo_root / "vocab" / "verb_alias.csv")
    return lexicon, aliases
