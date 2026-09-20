#!/usr/bin/env python3
"""Build same:true MC banks split by tier (machine JSON + human review CSV)."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
AQA = REPO_ROOT / "build" / "svo_aqa"
MC_DIR = AQA / "mc"
OUT_ROOT = AQA / "mc_same_true"
MATERIAL_JSON = AQA / "material.json"
VERB_JSON = AQA / "verb.json"

MC_SOURCES = (
    ("object_material_mc.json", "object_material_mc"),
    ("subject_material_mc.json", "subject_material_mc"),
    ("verb_mc.json", "verb_mc"),
)

TIER1 = "strict_minimal_pair"
TIER2 = "curated_distant_single"


def load_same_map() -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for path in (MATERIAL_JSON, VERB_JSON):
        for row in json.loads(path.read_text(encoding="utf-8")):
            out[row["audio_id"]] = bool(row.get("same"))
    return out


def load_clip_index() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for path in (MATERIAL_JSON, VERB_JSON):
        for row in json.loads(path.read_text(encoding="utf-8")):
            out[row["audio_id"]] = {
                "source_json": path.name,
                "same": row.get("same"),
                "caption": row.get("caption"),
                "seg_label": row.get("seg_label"),
                "verb": row.get("verb"),
                "verb_class": row.get("verb_class"),
                "subject": row.get("subject"),
                "object": row.get("object"),
                "subject_core": row.get("subject_core"),
                "object_core": row.get("object_core"),
                "subject_material": row.get("subject_material"),
                "object_material": row.get("object_material"),
            }
    return out


def load_all_questions() -> List[dict]:
    out: List[dict] = []
    for fname, task in MC_SOURCES:
        payload = json.loads((MC_DIR / fname).read_text(encoding="utf-8"))
        for q in payload["questions"]:
            q = dict(q)
            q["_source_file"] = fname
            if q.get("task") != task:
                q["task"] = task
            out.append(q)
    return out


def enrich_question(q: dict, clip_index: Dict[str, dict]) -> dict:
    aid = q["audio_id"]
    clip = clip_index.get(aid, {})
    row = dict(q)
    row["clip_meta"] = clip
    row["clip_same"] = clip.get("same")
    return row


def mat_label(v: str) -> str:
    return v.replace("_", " ")


def options_str(q: dict) -> str:
    parts = []
    for o in q["options"]:
        val = mat_label(o["value"]) if q["task"] != "verb_mc" else o["value"]
        parts.append(f"{o['label']}={val}")
    return " | ".join(parts)


def eval_role_label(q: dict) -> str:
    if q["task"] == "verb_mc":
        return "verb"
    if q["task"] == "object_material_mc":
        return "object_material"
    return "subject_material"


def human_row(q: dict) -> dict:
    ctx = q.get("prompt_context") or {}
    clip = q.get("clip_meta") or {}
    return {
        "question_id": q["question_id"],
        "tier": "tier1" if q["tier"] == TIER1 else "tier2",
        "task": q["task"],
        "group_id": q.get("group_id") or "",
        "audio_id": q["audio_id"],
        "wav": q.get("wav") or "",
        "caption": clip.get("caption") or "",
        "subject_core": ctx.get("subject_core") or "",
        "verb": ctx.get("verb") or "",
        "object_core": ctx.get("object_core") or "",
        "object_material_ctx": ctx.get("object_material") or "",
        "eval_target": eval_role_label(q),
        "gold": f"{q['gold_label']}={q['gold_value']}",
        "options": options_str(q),
        "verb_class": clip.get("verb_class") or "",
        "subject_material_clip": clip.get("subject_material") or "",
        "object_material_clip": clip.get("object_material") or "",
        "clip_same": q.get("clip_same"),
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def bank_payload(
    *,
    tier_key: str,
    tier_label: str,
    questions: List[dict],
    excluded: List[dict],
    same_map: Dict[str, bool],
) -> dict:
    by_task = defaultdict(list)
    for q in questions:
        by_task[q["task"]].append(q)

    task_counts = {k: len(v) for k, v in sorted(by_task.items())}
    clip_ids = sorted({q["audio_id"] for q in questions})

    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bank_id": f"mc_same_true_{tier_key}",
            "tier": tier_label,
            "filter": "clip_meta.same == true",
            "source_mc_dir": str(MC_DIR),
            "n_questions": len(questions),
            "n_clips": len(clip_ids),
            "n_by_task": task_counts,
            "n_excluded_same_false": len(excluded),
        },
        "questions": questions,
        "clip_ids": clip_ids,
        "excluded_question_ids": [q["question_id"] for q in excluded],
    }


def main() -> None:
    global AQA, MC_DIR, MATERIAL_JSON, VERB_JSON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=AQA,
        help="Directory containing human-annotated material.json and verb.json",
    )
    parser.add_argument(
        "--mc-dir",
        type=Path,
        default=None,
        help="MC drafts from build_aqascore_clusters.py (default: <annotations-dir>/mc)",
    )
    parser.add_argument("--material-json", type=Path, default=None)
    parser.add_argument("--verb-json", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <annotations-dir>/mc_same_true)",
    )
    args = parser.parse_args()

    AQA = args.annotations_dir
    MC_DIR = args.mc_dir or (AQA / "mc")
    MATERIAL_JSON = args.material_json or (AQA / "material.json")
    VERB_JSON = args.verb_json or (AQA / "verb.json")

    same_map = load_same_map()
    clip_index = load_clip_index()
    all_q = load_all_questions()

    kept: List[dict] = []
    excluded: List[dict] = []
    for q in all_q:
        same = same_map.get(q["audio_id"])
        eq = enrich_question(q, clip_index)
        if same is True:
            kept.append(eq)
        else:
            excluded.append(eq)

    tiers = {
        "tier1": (TIER1, [q for q in kept if q["tier"] == TIER1]),
        "tier2": (TIER2, [q for q in kept if q["tier"] == TIER2]),
    }

    out_dir = args.out_dir or (AQA / "mc_same_true")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filter": "same:true clips only",
        "source_total_questions": len(all_q),
        "kept_questions": len(kept),
        "excluded_questions": len(excluded),
        "excluded_clips": len({q["audio_id"] for q in excluded}),
        "tiers": {},
    }

    for tier_key, (tier_val, qs) in tiers.items():
        tier_dir = out_dir / tier_key
        tier_dir.mkdir(parents=True, exist_ok=True)

        ex_in_tier = [q for q in excluded if q["tier"] == tier_val]
        payload = bank_payload(
            tier_key=tier_key,
            tier_label=tier_val,
            questions=sorted(qs, key=lambda x: x["question_id"]),
            excluded=ex_in_tier,
            same_map=same_map,
        )

        # machine: combined bank
        bank_path = tier_dir / "bank.json"
        bank_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        # machine: by task (audit)
        by_task_dir = tier_dir / "by_task"
        by_task_dir.mkdir(exist_ok=True)
        task_groups = defaultdict(list)
        for q in qs:
            task_groups[q["task"]].append(q)
        for task, tqs in sorted(task_groups.items()):
            task_path = by_task_dir / f"{task}.json"
            task_path.write_text(
                json.dumps(
                    {
                        "meta": {
                            "bank_id": f"mc_same_true_{tier_key}",
                            "task": task,
                            "n_questions": len(tqs),
                        },
                        "questions": sorted(tqs, key=lambda x: x["question_id"]),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

        # human: review sheet
        human_rows = [human_row(q) for q in sorted(qs, key=lambda x: (x["task"], x["question_id"]))]
        write_csv(tier_dir / "review.csv", human_rows)

        # human: markdown summary
        md_lines = [
            f"# MC bank · same:true · {tier_key}",
            "",
            f"- tier: `{tier_val}`",
            f"- questions: **{len(qs)}**",
            f"- clips: **{len({q['audio_id'] for q in qs})}**",
            f"- by task: {dict(Counter(q['task'] for q in qs))}",
            "",
            "| # | question_id | task | audio_id | gold | options |",
            "|---|-------------|------|----------|------|---------|",
        ]
        for i, r in enumerate(human_rows, 1):
            md_lines.append(
                f"| {i} | `{r['question_id']}` | {r['task']} | `{r['audio_id']}` | {r['gold']} | {r['options']} |"
            )
        (tier_dir / "review.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        manifest["tiers"][tier_key] = {
            "tier": tier_val,
            "n_questions": len(qs),
            "n_clips": len({q["audio_id"] for q in qs}),
            "n_by_task": dict(Counter(q["task"] for q in qs)),
            "bank_json": str(bank_path),
            "review_csv": str(tier_dir / "review.csv"),
            "review_md": str(tier_dir / "review.md"),
        }

    # excluded audit
    ex_path = out_dir / "excluded_same_false.json"
    ex_path.write_text(
        json.dumps(
            {
                "meta": {
                    "reason": "clip same:false — ambiguous subject/object material attribution",
                    "n_questions": len(excluded),
                    "n_clips": len({q["audio_id"] for q in excluded}),
                },
                "questions": sorted(excluded, key=lambda x: x["question_id"]),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest["excluded_path"] = str(ex_path)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
