#!/usr/bin/env python3
"""Cluster aqascore clips into controlled groups and draft 4-choice MC questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from itertools import combinations

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARK_ROOT / "lib"))

from core_material_leak import core_leaks_material  # noqa: E402

AQA_DIR = REPO_ROOT / "build" / "svo_aqa"
MATERIAL_JSON = AQA_DIR / "material.json"
VERB_JSON = AQA_DIR / "verb.json"
OUT_CLUSTERS = AQA_DIR / "clusters"
OUT_CLIPS = AQA_DIR / "clips"
OUT_MC = AQA_DIR / "mc"

LETTERS = ("A", "B", "C", "D")

# 听感相近材质族：同族内 NOT distant，不可进同一 minimal-pair 组。
# rubber/wax/leather/paper/wood 偏软/有机；plastic/glass/metal 有时互混；
# textile_fibrous、clay 各成一族；两种 foam 各自独立。
MATERIAL_NEAR_FAMILIES: Tuple[frozenset[str], ...] = (
    frozenset({"rubber", "wax", "leather", "paper", "wood"}),
    frozenset({"plastic", "glass", "metal"}),
    frozenset({"textile_fibrous"}),
    frozenset({"clay"}),
    frozenset({"foam_lather"}),
    frozenset({"foam_solid"}),
)

COMPACT_MATERIALS_PATH = REPO_ROOT / "vocab" / "asmr_material_classes_compact_en.csv"

VERB_CLASS_PATH = REPO_ROOT / "vocab" / "asmr_verb_classes_en.csv"


def norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = re.sub(r"\s+", " ", str(s).strip().lower())
    return t or None


def slug_part(s: Optional[str]) -> str:
    if not s:
        return "_"
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "_"


def stable_shuffle(items: List[str], seed: str) -> List[str]:
    keyed = [(hashlib.md5(f"{seed}:{x}".encode()).hexdigest(), x) for x in items]
    keyed.sort(key=lambda t: t[0])
    return [x for _, x in keyed]


def load_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_verb_to_class() -> Dict[str, str]:
    import csv

    out: Dict[str, str] = {}
    with VERB_CLASS_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cls = (row.get("class") or "").strip()
            for v in (row.get("verbs") or "").split(";"):
                v = v.strip()
                if v:
                    out[v] = cls
    return out


def clip_row(row: dict) -> dict:
    keys = (
        "audio_id", "wav", "caption", "seg_label", "verb", "verb_class",
        "subject", "object", "subject_core", "object_core",
        "subject_material", "object_material",
    )
    return {k: row.get(k) for k in keys}


def group_id_from_key(prefix: str, key: Tuple) -> str:
    return prefix + "__" + "__".join(slug_part(x) for x in key)


def load_compact_material_pool() -> Set[str]:
    import csv

    out: Set[str] = set()
    with COMPACT_MATERIALS_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cls = (row.get("class") or "").strip()
            if cls:
                out.add(cls)
    return out


def material_family(mat: str) -> frozenset[str]:
    for fam in MATERIAL_NEAR_FAMILIES:
        if mat in fam:
            return fam
    return frozenset({mat})


def materials_distant(a: str, b: str) -> bool:
    if a == b:
        return False
    return material_family(a) != material_family(b)


def maximal_distant_subsets(mats: Set[str]) -> List[Set[str]]:
    """Maximal subsets where every pair of materials is acoustically distant."""
    items = sorted(mats)
    n = len(items)
    maximal: List[Set[str]] = []
    for size in range(n, 1, -1):
        for combo in combinations(items, size):
            subset = set(combo)
            if not all(materials_distant(a, b) for a in subset for b in subset if a != b):
                continue
            if any(subset < existing for existing in maximal):
                continue
            maximal = [existing for existing in maximal if not existing < subset]
            maximal.append(subset)
    return maximal


def near_neighbor_materials(mat: str) -> Set[str]:
    """Same-family materials (for padding exclusion)."""
    return set(material_family(mat))


def pad_materials_distant(
    gold: str,
    group_values: Set[str],
    pool: Set[str],
    k: int,
    seed: str,
) -> List[str]:
    """Pad MC with non-neighbor materials; stable pseudo-random order."""
    if k <= 0:
        return []
    blocked = group_values | near_neighbor_materials(gold)
    candidates = [m for m in sorted(pool) if m not in blocked]
    candidates = stable_shuffle(candidates, f"{seed}:pad")
    return candidates[:k]


def pad_verbs_distant(
    gold: str,
    verb_to_class: Dict[str, str],
    pool: Set[str],
    k: int,
    seed: str,
) -> List[str]:
    """Prefer verbs from a different verb_class than gold."""
    if k <= 0:
        return []
    cls = verb_to_class.get(gold)
    same_class = {v for v in pool if v != gold and verb_to_class.get(v) == cls}
    blocked = {gold} | same_class
    candidates = [v for v in sorted(pool) if v not in blocked]
    candidates = stable_shuffle(candidates, f"{seed}:vpad")
    if len(candidates) < k:
        for v in sorted(pool):
            if v != gold and v not in candidates:
                candidates.append(v)
    return candidates[:k]


def near_verbs(verb: str, verb_to_class: Dict[str, str], pool: Set[str], k: int) -> List[str]:
    cls = verb_to_class.get(verb)
    same = [v for v in sorted(pool) if v != verb and verb_to_class.get(v) == cls]
    other = [v for v in sorted(pool) if v != verb and verb_to_class.get(v) != cls]
    out = same + other
    return out[:k]


def pick_options(
    gold: str,
    group_values: Set[str],
    *,
    kind: str,
    seed: str,
    verb_to_class: Dict[str, str],
    material_pool: Set[str],
    verb_pool: Set[str],
    max_options: int = 4,
    verb_pad: str = "near",
) -> Tuple[List[dict], str]:
    """Return options list and gold letter label."""
    values = [gold] + [v for v in sorted(group_values) if v != gold]
    if len(values) < max_options:
        if kind == "material":
            pad = pad_materials_distant(
                gold, group_values, material_pool, max_options - len(values), seed
            )
        else:
            if verb_pad == "distant":
                pad = pad_verbs_distant(
                    gold, verb_to_class, verb_pool, max_options - len(values), seed
                )
            else:
                pad = near_verbs(gold, verb_to_class, verb_pool, max_options - len(values))
        for p in pad:
            if p not in values:
                values.append(p)
    values = values[:max_options]
    values = stable_shuffle(values, seed)
    labels = LETTERS[: len(values)]
    gold_label = labels[values.index(gold)]
    options = []
    for lab, val in zip(labels, values):
        if val == gold:
            src = "target"
        elif val in group_values:
            src = "same_group"
        elif kind == "material":
            src = "distant_pool"
        elif verb_pad == "distant":
            src = "distant_pool"
        else:
            src = "near_pool"
        opt: dict = {"label": lab, "value": val, "source": src}
        if kind == "verb":
            opt["verb_class"] = verb_to_class.get(val)
        options.append(opt)
    return options, gold_label


def _material_groups_from_bucket(
    *,
    prefix: str,
    key: Tuple,
    members: List[dict],
    mat_field: str,
    task: str,
    group_key_fields: Dict[str, Any],
    distinct_field: str,
) -> Dict[str, dict]:
    """Split bucket into groups whose material labels are pairwise distant."""
    mats = {m[mat_field] for m in members}
    subsets = maximal_distant_subsets(mats)
    out: Dict[str, dict] = {}
    for idx, mat_subset in enumerate(sorted(subsets, key=lambda s: sorted(s))):
        sub_members = [m for m in members if m[mat_field] in mat_subset]
        actual = sorted({m[mat_field] for m in sub_members})
        if len(actual) < 2:
            continue
        gid = group_id_from_key(prefix, key)
        if len(subsets) > 1:
            gid += "__d" + slug_part("-".join(actual))
        out[gid] = {
            "group_id": gid,
            "task": task,
            "group_key": dict(group_key_fields),
            distinct_field: actual,
            "material_distant_only": True,
            "n_clips": len(sub_members),
            "members": [clip_row(m) for m in sub_members],
        }
    return out


def cluster_object_material(rows: Sequence[dict]) -> Dict[str, dict]:
    buckets: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        om = r.get("object_material")
        if not om:
            continue
        key = (norm(r.get("subject_core")), r.get("verb"), norm(r.get("object_core")))
        buckets[key].append(r)

    groups: Dict[str, dict] = {}
    for key, members in buckets.items():
        groups.update(
            _material_groups_from_bucket(
                prefix="om",
                key=key,
                members=members,
                mat_field="object_material",
                task="object_material_mc",
                group_key_fields={
                    "subject_core": key[0],
                    "verb": key[1],
                    "object_core": key[2],
                },
                distinct_field="distinct_object_materials",
            )
        )
    return groups


def cluster_subject_material(rows: Sequence[dict]) -> Dict[str, dict]:
    buckets: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        sm = r.get("subject_material")
        if not sm:
            continue
        key = (
            norm(r.get("subject_core")),
            r.get("verb"),
            norm(r.get("object_core")),
            r.get("object_material"),
        )
        buckets[key].append(r)

    groups: Dict[str, dict] = {}
    for key, members in buckets.items():
        groups.update(
            _material_groups_from_bucket(
                prefix="sm",
                key=key,
                members=members,
                mat_field="subject_material",
                task="subject_material_mc",
                group_key_fields={
                    "subject_core": key[0],
                    "verb": key[1],
                    "object_core": key[2],
                    "object_material": key[3],
                },
                distinct_field="distinct_subject_materials",
            )
        )
    return groups


def cluster_verb(rows: Sequence[dict]) -> Dict[str, dict]:
    buckets: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        key = (norm(r.get("subject_core")), norm(r.get("object_core")))
        buckets[key].append(r)

    groups: Dict[str, dict] = {}
    for key, members in buckets.items():
        verbs = {m["verb"] for m in members}
        if len(verbs) < 2:
            continue
        gid = group_id_from_key("vb", key)
        groups[gid] = {
            "group_id": gid,
            "task": "verb_mc",
            "group_key": {
                "subject_core": key[0],
                "object_core": key[1],
            },
            "distinct_verbs": sorted(verbs),
            "n_clips": len(members),
            "members": [clip_row(m) for m in members],
        }
    return groups


def build_mc_questions(
    groups: Dict[str, dict],
    *,
    task: str,
    tier: str,
    verb_to_class: Dict[str, str],
    material_pool: Set[str],
    verb_pool: Set[str],
) -> List[dict]:
    questions: List[dict] = []
    for gid in sorted(groups):
        g = groups[gid]
        if task == "object_material_mc":
            kind = "material"
            value_key = "object_material"
            group_vals = set(g["distinct_object_materials"])
            prompt_context = {
                "subject_core": g["group_key"]["subject_core"],
                "verb": g["group_key"]["verb"],
                "object_core": g["group_key"]["object_core"],
            }
            eval_role = "object"
        elif task == "subject_material_mc":
            kind = "material"
            value_key = "subject_material"
            group_vals = set(g["distinct_subject_materials"])
            prompt_context = {
                "subject_core": g["group_key"]["subject_core"],
                "verb": g["group_key"]["verb"],
                "object_core": g["group_key"]["object_core"],
                "object_material": g["group_key"]["object_material"],
            }
            eval_role = "subject"
        else:
            kind = "verb"
            value_key = "verb"
            group_vals = set(g["distinct_verbs"])
            prompt_context = {
                "subject_core": g["group_key"]["subject_core"],
                "object_core": g["group_key"]["object_core"],
            }
            eval_role = None

        for inst, member in enumerate(g["members"]):
            gold = member[value_key]
            if not gold:
                continue
            qid = f"{gid}__inst{inst}"
            options, gold_label = pick_options(
                gold,
                group_vals,
                kind=kind,
                seed=qid,
                verb_to_class=verb_to_class,
                material_pool=material_pool,
                verb_pool=verb_pool,
            )
            leaks_obj, _, _ = core_leaks_material(member.get("object_core"))
            leaks_sub, _, _ = core_leaks_material(member.get("subject_core"))
            questions.append({
                "question_id": qid,
                "group_id": gid,
                "task": task,
                "tier": tier,
                "audio_id": member["audio_id"],
                "wav": member["wav"],
                "prompt_context": prompt_context,
                "eval_role": eval_role,
                "options": options,
                "gold_label": gold_label,
                "gold_value": gold,
                "core_leakage": {
                    "subject_core": leaks_sub,
                    "object_core": leaks_obj,
                },
                "clip": member,
            })
    return questions


def tier1_member_ids(groups: Dict[str, dict]) -> Set[str]:
    return {m["audio_id"] for g in groups.values() for m in g["members"]}


def build_tier2_material_mc(
    rows: Sequence[dict],
    *,
    task: str,
    mat_field: str,
    exclude_ids: Set[str],
    verb_to_class: Dict[str, str],
    material_pool: Set[str],
    verb_pool: Set[str],
) -> Tuple[List[dict], List[dict]]:
    """Single-clip MC: gold from label, distractors from distant pool only."""
    questions: List[dict] = []
    clips: List[dict] = []
    prefix = "t2_om" if task == "object_material_mc" else "t2_sm"

    for row in rows:
        gold = row.get(mat_field)
        if not gold or row["audio_id"] in exclude_ids:
            continue
        member = clip_row(row)
        if task == "object_material_mc":
            prompt_context = {
                "subject_core": norm(row.get("subject_core")),
                "verb": row.get("verb"),
                "object_core": norm(row.get("object_core")),
            }
            eval_role = "object"
        else:
            prompt_context = {
                "subject_core": norm(row.get("subject_core")),
                "verb": row.get("verb"),
                "object_core": norm(row.get("object_core")),
                "object_material": row.get("object_material"),
            }
            eval_role = "subject"

        qid = f"{prefix}__{row['audio_id']}"
        options, gold_label = pick_options(
            gold,
            {gold},
            kind="material",
            seed=qid,
            verb_to_class=verb_to_class,
            material_pool=material_pool,
            verb_pool=verb_pool,
        )
        leaks_obj, _, _ = core_leaks_material(member.get("object_core"))
        leaks_sub, _, _ = core_leaks_material(member.get("subject_core"))
        questions.append({
            "question_id": qid,
            "group_id": None,
            "task": task,
            "tier": "curated_distant_single",
            "audio_id": row["audio_id"],
            "wav": row["wav"],
            "prompt_context": prompt_context,
            "eval_role": eval_role,
            "options": options,
            "gold_label": gold_label,
            "gold_value": gold,
            "core_leakage": {
                "subject_core": leaks_sub,
                "object_core": leaks_obj,
            },
            "clip": member,
        })
        clips.append({**member, "question_id": qid, "tier": "curated_distant_single"})

    return questions, clips


def build_tier2_verb_mc(
    rows: Sequence[dict],
    *,
    exclude_ids: Set[str],
    verb_to_class: Dict[str, str],
    material_pool: Set[str],
    verb_pool: Set[str],
) -> Tuple[List[dict], List[dict]]:
    questions: List[dict] = []
    clips: List[dict] = []
    for row in rows:
        gold = row.get("verb")
        if not gold or row["audio_id"] in exclude_ids:
            continue
        member = clip_row(row)
        prompt_context = {
            "subject_core": norm(row.get("subject_core")),
            "object_core": norm(row.get("object_core")),
        }
        qid = f"t2_vb__{row['audio_id']}"
        options, gold_label = pick_options(
            gold,
            {gold},
            kind="verb",
            seed=qid,
            verb_to_class=verb_to_class,
            material_pool=material_pool,
            verb_pool=verb_pool,
            verb_pad="distant",
        )
        leaks_obj, _, _ = core_leaks_material(member.get("object_core"))
        leaks_sub, _, _ = core_leaks_material(member.get("subject_core"))
        questions.append({
            "question_id": qid,
            "group_id": None,
            "task": "verb_mc",
            "tier": "curated_distant_single",
            "audio_id": row["audio_id"],
            "wav": row["wav"],
            "prompt_context": prompt_context,
            "eval_role": None,
            "options": options,
            "gold_label": gold_label,
            "gold_value": gold,
            "core_leakage": {
                "subject_core": leaks_sub,
                "object_core": leaks_obj,
            },
            "clip": member,
        })
        clips.append({**member, "question_id": qid, "tier": "curated_distant_single"})
    return questions, clips


def clips_from_groups(groups: Dict[str, dict]) -> List[dict]:
    by_id: Dict[str, dict] = {}
    for gid, g in groups.items():
        for m in g["members"]:
            aid = m["audio_id"]
            if aid not in by_id:
                by_id[aid] = {**m, "group_ids": []}
            if gid not in by_id[aid]["group_ids"]:
                by_id[aid]["group_ids"].append(gid)
    return [by_id[k] for k in sorted(by_id)]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-json", type=Path, default=MATERIAL_JSON)
    parser.add_argument("--verb-json", type=Path, default=VERB_JSON)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=AQA_DIR,
        help="Output root containing clusters/, clips/, and mc/",
    )
    args = parser.parse_args()

    out_clusters = args.out_dir / "clusters"
    out_clips = args.out_dir / "clips"
    out_mc = args.out_dir / "mc"

    material_rows = load_json(args.material_json)
    verb_rows = load_json(args.verb_json)
    verb_to_class = load_verb_to_class()

    material_pool = load_compact_material_pool()
    material_pool |= {
        m for r in material_rows for m in (r.get("subject_material"), r.get("object_material")) if m
    }
    verb_pool = {r["verb"] for r in verb_rows if r.get("verb")}

    om_groups = cluster_object_material(material_rows)
    sm_groups = cluster_subject_material(material_rows)
    vb_groups = cluster_verb(verb_rows)

    om_t1_ids = tier1_member_ids(om_groups)
    sm_t1_ids = tier1_member_ids(sm_groups)

    vb_t1_ids = tier1_member_ids(vb_groups)

    om_mc_t1 = build_mc_questions(
        om_groups, task="object_material_mc", tier="strict_minimal_pair",
        verb_to_class=verb_to_class, material_pool=material_pool, verb_pool=verb_pool,
    )
    sm_mc_t1 = build_mc_questions(
        sm_groups, task="subject_material_mc", tier="strict_minimal_pair",
        verb_to_class=verb_to_class, material_pool=material_pool, verb_pool=verb_pool,
    )
    om_mc_t2, om_clips_t2 = build_tier2_material_mc(
        material_rows, task="object_material_mc", mat_field="object_material",
        exclude_ids=om_t1_ids, verb_to_class=verb_to_class,
        material_pool=material_pool, verb_pool=verb_pool,
    )
    sm_mc_t2, sm_clips_t2 = build_tier2_material_mc(
        material_rows, task="subject_material_mc", mat_field="subject_material",
        exclude_ids=sm_t1_ids, verb_to_class=verb_to_class,
        material_pool=material_pool, verb_pool=verb_pool,
    )
    vb_mc_t1 = build_mc_questions(
        vb_groups, task="verb_mc", tier="strict_minimal_pair",
        verb_to_class=verb_to_class, material_pool=material_pool, verb_pool=verb_pool,
    )
    vb_mc_t2, vb_clips_t2 = build_tier2_verb_mc(
        verb_rows, exclude_ids=vb_t1_ids, verb_to_class=verb_to_class,
        material_pool=material_pool, verb_pool=verb_pool,
    )

    om_mc = om_mc_t1 + om_mc_t2
    sm_mc = sm_mc_t1 + sm_mc_t2
    vb_mc = vb_mc_t1 + vb_mc_t2

    meta = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_material_json": str(args.material_json),
        "source_verb_json": str(args.verb_json),
        "material_near_families": [sorted(f) for f in MATERIAL_NEAR_FAMILIES],
        "cluster_rule": "group内材质两两 distant（不同 near family）",
        "tier1_rule": "strict_minimal_pair：同 group_key 内 ≥2 distant 材质 clip",
        "tier2_rule": "curated_distant_single：未进 tier1 的单 clip，distant 补位 4 选 1",
        "verb_tier2_pad": "distractors 优先来自不同 verb_class",
    }

    write_json(out_clusters / "object_material_groups.json", {"meta": meta, "groups": list(om_groups.values())})
    write_json(out_clusters / "subject_material_groups.json", {"meta": meta, "groups": list(sm_groups.values())})
    write_json(out_clusters / "verb_groups.json", {"meta": meta, "groups": list(vb_groups.values())})

    write_json(out_clips / "object_material_clips.json", {"meta": meta, "data": clips_from_groups(om_groups)})
    write_json(out_clips / "subject_material_clips.json", {"meta": meta, "data": clips_from_groups(sm_groups)})
    write_json(out_clips / "verb_clips.json", {"meta": meta, "data": clips_from_groups(vb_groups)})

    write_json(out_mc / "object_material_mc.json", {
        "meta": {**meta, "n_questions": len(om_mc), "n_tier1": len(om_mc_t1), "n_tier2": len(om_mc_t2)},
        "questions": om_mc,
    })
    write_json(out_mc / "subject_material_mc.json", {
        "meta": {**meta, "n_questions": len(sm_mc), "n_tier1": len(sm_mc_t1), "n_tier2": len(sm_mc_t2)},
        "questions": sm_mc,
    })
    write_json(out_mc / "object_material_mc_tier2.json", {
        "meta": {**meta, "n_questions": len(om_mc_t2)}, "questions": om_mc_t2,
    })
    write_json(out_mc / "subject_material_mc_tier2.json", {
        "meta": {**meta, "n_questions": len(sm_mc_t2)}, "questions": sm_mc_t2,
    })
    write_json(out_mc / "verb_mc.json", {
        "meta": {**meta, "n_questions": len(vb_mc), "n_tier1": len(vb_mc_t1), "n_tier2": len(vb_mc_t2)},
        "questions": vb_mc,
    })
    write_json(out_mc / "verb_mc_tier2.json", {
        "meta": {**meta, "n_questions": len(vb_mc_t2)}, "questions": vb_mc_t2,
    })

    write_json(out_clips / "object_material_clips_tier2.json", {"meta": meta, "data": om_clips_t2})
    write_json(out_clips / "subject_material_clips_tier2.json", {"meta": meta, "data": sm_clips_t2})
    write_json(out_clips / "verb_clips_tier2.json", {"meta": meta, "data": vb_clips_t2})

    n_t1 = len(om_mc_t1) + len(sm_mc_t1) + len(vb_mc_t1)
    n_t2 = len(om_mc_t2) + len(sm_mc_t2) + len(vb_mc_t2)
    n_all = n_t1 + n_t2

    print(f"object_material tier1: {len(om_groups)} groups, {len(om_t1_ids)} clips, {len(om_mc_t1)} MC")
    print(f"object_material tier2: {len(om_clips_t2)} clips, {len(om_mc_t2)} MC")
    print(f"object_material total: {len(om_mc)} MC")
    print(f"subject_material tier1: {len(sm_groups)} groups, {len(sm_t1_ids)} clips, {len(sm_mc_t1)} MC")
    print(f"subject_material tier2: {len(sm_clips_t2)} clips, {len(sm_mc_t2)} MC")
    print(f"subject_material total: {len(sm_mc)} MC")
    print(f"verb tier1: {len(vb_groups)} groups, {len(vb_t1_ids)} clips, {len(vb_mc_t1)} MC")
    print(f"verb tier2: {len(vb_clips_t2)} clips, {len(vb_mc_t2)} MC")
    print(f"verb total: {len(vb_mc)} MC")
    print(f"tier1 total: {n_t1}")
    print(f"tier2 total: {n_t2}")
    print(f"all MC total: {n_all}")


if __name__ == "__main__":
    main()
