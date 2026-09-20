"""Detect subject_core / object_core nouns that leak material class labels."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIAS = REPO_ROOT / "vocab" / "material_alias.csv"

# Core head nouns that strongly imply a material but may be absent from alias.csv.
# Exact cores kept despite partial material-token overlap (manual review).
CORE_LEAK_ALLOWLIST = frozenset({"sponge", "wooden surface"})

EXTRA_MATERIAL_CORES = {
    "ice": "glass",
    "styrofoam": "foam_solid",
    "silicone": "rubber",
    "chalk": "ceramic",
    "marble": "stone",
    "granite": "stone",
    "beeswax": "wax",
    "cardboard": "paper",
}


def norm_phrase(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _singularize(word: str) -> str:
    w = word.lower()
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 2:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 1:
        return w[:-1]
    return w


def load_material_aliases(path: Path = DEFAULT_ALIAS) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            alias = norm_phrase(row.get("alias"))
            gid = (row.get("group_id") or "").strip()
            if alias and gid:
                out[alias] = gid
    return out


def core_leaks_material(
    core: Optional[str],
    *,
    alias_to_group: Optional[dict[str, str]] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Return (leaks, reason, inferred_material_group).

    A core leaks if it equals or contains a material alias (e.g. cork → wood).
    """
    if core is None:
        return False, None, None
    text = norm_phrase(core)
    if not text:
        return False, None, None

    if text in CORE_LEAK_ALLOWLIST:
        return False, None, None

    aliases = alias_to_group or load_material_aliases()

    if "material" in text.split():
        return True, "contains_word_material", None

    if text in aliases:
        return True, f"exact_alias:{text}", aliases[text]

    if text in EXTRA_MATERIAL_CORES:
        gid = EXTRA_MATERIAL_CORES[text]
        return True, f"extra_core:{text}", gid

    for alias, gid in sorted(aliases.items(), key=lambda x: -len(x[0])):
        if " " in alias and alias in text:
            return True, f"subphrase_alias:{alias}", gid

    for token in text.split():
        for form in (token, _singularize(token)):
            if form in aliases:
                return True, f"token_alias:{form}", aliases[form]
            if form in EXTRA_MATERIAL_CORES:
                return True, f"extra_token:{form}", EXTRA_MATERIAL_CORES[form]

    return False, None, None


def null_leaking_cores(row: dict, *, alias_to_group: Optional[dict[str, str]] = None) -> dict:
    """Return row copy with leaking subject_core / object_core set to null."""
    out = dict(row)
    aliases = alias_to_group or load_material_aliases()
    for role in ("subject", "object"):
        key = f"{role}_core"
        leaks, _, _ = core_leaks_material(out.get(key), alias_to_group=aliases)
        if leaks:
            out[key] = None
    return out
