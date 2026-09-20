#!/usr/bin/env python3
"""Prepare a DeepASMR-NSpeech staging folder for Hugging Face upload.

Layout:
  DeepASMR-NSpeech-dataset/
    README.md
    train/train_label.json + train/audio/...
    test/test_label.json   + test/audio/...
    SVO-AQA/test_subset/      # 431-question benchmark only

Usage:
  python3 scripts/prepare_deepasmr_nspeech_hf.py --labels-only
  python3 scripts/prepare_deepasmr_nspeech_hf.py --copy-audio --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "DeepASMR-NSpeech-dataset"
TRAIN_LABEL_SRC = ROOT / "train_label.json"
TEST_LABEL_SRC = ROOT / "test_label.json"
TRAIN_AUDIO_PREFIX = ROOT / "data" / "train" / "audio"
MC_SAME_TRUE_SRC = ROOT / "build" / "svo_aqa" / "mc_same_true"
SVO_AQA_REL = "SVO-AQA/test_subset"

TRAIN_REL_PREFIX = "train/audio"
TEST_REL_PREFIX = "test/audio"

# 不进入公开 release 的字段（author 转为 creator_id；material 仅保留在 SVO-AQA MC 题中）
LABEL_DROP_KEYS = frozenset({"seg_label", "title", "author", "subject_material", "object_material"})
CREATOR_MAP_PATH = ROOT / "build" / "private" / "creator_map.json"
VOCAB_REF_PATH = OUT_ROOT / "vocab" / "closed_vocab_reference.md"
VERB_CLASSES_CSV = ROOT / "vocab" / "asmr_verb_classes_en.csv"
MATERIAL_CLASSES_CSV = ROOT / "vocab" / "asmr_material_classes_compact_en.csv"
VERB_DISAMBIG_TXT = ROOT / "vocab" / "stage2_class_disambiguation.txt"
DATASET_CARD_SRC = ROOT / "docs" / "dataset_card.md"
SVO_AQA_SUBSET_SRC = ROOT / "benchmark" / "svo_aqa" / "audio_dependent_subset.v1.json"
QUESTION_DROP_KEYS = frozenset({
    "tier", "clip_meta", "_source_file", "core_leakage",
    "clip_same", "group_id", "eval_role",
})
CLIP_DROP_KEYS = frozenset({"seg_label", "wav", "audio_id", "subject_material", "object_material"})
OPTION_DROP_KEYS = frozenset({"source"})

PROMPT_TEMPLATES = {
    "verb_mc": (
        "Subject: {subject_core}\n"
        "Object: {object_core}\n"
        "Which action best matches the sound?\n"
        "{options}\n"
        "Answer with a single letter ({letters}) only."
    ),
    "object_material_mc": (
        "Subject: {subject_core}\n"
        "Action: {verb}\n"
        "Object: {object_core}\n"
        "What is the object's material?\n"
        "{options}\n"
        "Answer with a single letter ({letters}) only."
    ),
    "subject_material_mc": (
        "Subject: {subject_core}\n"
        "Action: {verb}\n"
        "Object: {object_core}{object_material_suffix}\n"
        "What is the subject's material?\n"
        "{options}\n"
        "Answer with a single letter ({letters}) only."
    ),
}


def mat_label(value: str) -> str:
    return value.replace("_", " ")


def build_mc_prompt(q: dict) -> str:
    """Render the public prompt included with each SVO-AQA question."""
    ctx = q["prompt_context"]
    opts = q["options"]
    opt_lines = "\n".join(
        f"({o['label']}) {mat_label(o['value']) if q['task'] != 'verb_mc' else o['value']}"
        for o in opts
    )
    letters = "/".join(o["label"] for o in opts)
    sub = ctx.get("subject_core") or "item"
    obj = ctx.get("object_core") or "surface"
    tmpl = PROMPT_TEMPLATES[q["task"]]
    om = ctx.get("object_material")
    om_suffix = f" ({mat_label(om)})" if om else ""
    return tmpl.format(
        subject_core=sub,
        object_core=obj,
        verb=ctx.get("verb", ""),
        object_material_suffix=om_suffix,
        options=opt_lines,
        letters=letters,
    )


def train_rel_wav(src: Path) -> str:
    try:
        rel = src.relative_to(TRAIN_AUDIO_PREFIX)
        return f"{TRAIN_REL_PREFIX}/{rel.as_posix()}"
    except ValueError:
        parts = list(src.parts)
        if "splitted" in parts:
            rel = Path(*parts[parts.index("splitted") + 1 :])
            return f"{TRAIN_REL_PREFIX}/{rel.as_posix()}"
        return f"{TRAIN_REL_PREFIX}/{src.name}"


def test_rel_wav(src: Path) -> str:
    return f"{TEST_REL_PREFIX}/{src.name}"


def rewrite_wav_field(obj: Any, mapper: Callable[[Path], str]) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "wav" and isinstance(v, str) and v.strip():
                out[k] = mapper(Path(v.strip()))
            else:
                out[k] = rewrite_wav_field(v, mapper)
        return out
    if isinstance(obj, list):
        return [rewrite_wav_field(x, mapper) for x in obj]
    return obj


def load_label_rows(path: Path) -> Tuple[dict, List[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("data") or [])
    return payload, rows


def build_creator_id_map(*label_paths: Path) -> Dict[str, str]:
    """Stable anonymized IDs: creator_01 .. creator_N (sorted by author name)."""
    authors: set[str] = set()
    for path in label_paths:
        for row in load_label_rows(path)[1]:
            author = (row.get("author") or "").strip()
            if author:
                authors.add(author)
    ordered = sorted(authors)
    return {name: f"creator_{i:02d}" for i, name in enumerate(ordered, start=1)}


def write_creator_map(creator_map: Dict[str, str]) -> Path:
    CREATOR_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "Internal only — do NOT upload to Hugging Face.",
        "n_creators": len(creator_map),
        "creator_id_to_author": {v: k for k, v in sorted(creator_map.items(), key=lambda x: x[1])},
    }
    CREATOR_MAP_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return CREATOR_MAP_PATH


def export_label_json(
    src: Path,
    out: Path,
    mapper: Callable[[Path], str],
    creator_map: Dict[str, str],
) -> dict:
    payload, rows = load_label_rows(src)
    new_rows = []
    missing = 0
    for row in rows:
        wav = (row.get("wav") or "").strip()
        if not wav:
            missing += 1
            continue
        src_wav = Path(wav)
        new_row = {k: v for k, v in row.items() if k not in LABEL_DROP_KEYS}
        author = (row.get("author") or "").strip()
        new_row["creator_id"] = creator_map.get(author, "creator_unknown")
        new_row["wav"] = mapper(src_wav)
        new_rows.append(new_row)
        if not src_wav.is_file():
            missing += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"data": new_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        output_name = out.relative_to(OUT_ROOT).as_posix()
    except ValueError:
        output_name = out.name
    return {"file": output_name, "n_rows": len(new_rows), "missing_audio": missing}


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def materialize_train_audio(rows: Iterable[dict], *, copy: bool) -> dict:
    n_ok, n_miss = 0, 0
    for row in rows:
        src = Path((row.get("wav") or "").strip())
        if not src.is_file():
            n_miss += 1
            continue
        rel = train_rel_wav(src)
        dst = OUT_ROOT / rel
        link_or_copy(src, dst, copy=copy)
        n_ok += 1
    return {"linked_or_copied": n_ok, "missing": n_miss}


def materialize_test_audio(rows: Iterable[dict], *, copy: bool) -> dict:
    n_ok, n_miss = 0, 0
    for row in rows:
        src = Path((row.get("wav") or "").strip())
        if not src.is_file():
            n_miss += 1
            continue
        rel = test_rel_wav(src)
        dst = OUT_ROOT / rel
        link_or_copy(src, dst, copy=copy)
        n_ok += 1
    return {"linked_or_copied": n_ok, "missing": n_miss}


def clean_question(q: dict) -> dict:
    out = {k: v for k, v in q.items() if k not in QUESTION_DROP_KEYS}
    out["options"] = [
        {k: v for k, v in o.items() if k not in OPTION_DROP_KEYS}
        for o in (out.get("options") or [])
    ]
    if isinstance(out.get("clip"), dict):
        clip = {k: v for k, v in out["clip"].items() if k not in CLIP_DROP_KEYS}
        out["clip"] = clip
    wav = (out.get("wav") or "").strip()
    if wav:
        out["wav"] = test_rel_wav(Path(wav))
    out["prompt"] = build_mc_prompt(out)
    return out


def export_svo_aqa_bank(src_root: Path, out_path: Path) -> dict:
    """Merge tier1+tier2 into one bank.json (431 questions, release schema)."""
    questions: List[dict] = []
    for tier in ("tier1", "tier2"):
        bank_path = src_root / tier / "bank.json"
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
        questions.extend(payload.get("questions") or [])

    cleaned = [clean_question(q) for q in questions]
    audio_ids = {q.get("audio_id") for q in cleaned if q.get("audio_id")}
    by_task: Dict[str, int] = {}
    for q in cleaned:
        task = q.get("task") or "unknown"
        by_task[task] = by_task.get(task, 0) + 1

    payload = {
        "meta": {
            "n_questions": len(cleaned),
            "n_clips": len(audio_ids),
            "n_by_task": by_task,
            "prompt_templates": PROMPT_TEMPLATES,
            "usage": (
                "For each question: load audio from wav (relative to dataset root), "
                "send audio + prompt to your model, compare predicted letter to gold_label."
            ),
        },
        "questions": cleaned,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "file": out_path.relative_to(OUT_ROOT).as_posix(),
        "n_questions": len(cleaned),
        "n_clips": len(audio_ids),
        "n_by_task": by_task,
    }


def load_disambiguation_rules(path: Path) -> Dict[str, Tuple[str, List[str]]]:
    """Parse stage2_class_disambiguation.txt -> {class_id: (title, [rules])}."""
    out: Dict[str, Tuple[str, List[str]]] = {}
    if not path.is_file():
        return out
    current_id: Optional[str] = None
    title = ""
    rules: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " — " in line and line.endswith(":"):
            if current_id:
                out[current_id] = (title, rules)
            head, title = line[:-1].split(" — ", 1)
            current_id = head.strip()
            rules = []
            continue
        if line.startswith("- ") and current_id:
            rules.append(line[2:].strip())
    if current_id:
        out[current_id] = (title, rules)
    return out


def generate_vocab_reference() -> Path:
    """Write tree-style closed-vocab reference for Hugging Face (not full CSV/prompts)."""
    verb_rows = list(csv.DictReader(VERB_CLASSES_CSV.open(encoding="utf-8", newline="")))
    mat_rows = list(csv.DictReader(MATERIAL_CLASSES_CSV.open(encoding="utf-8", newline="")))
    disambig = load_disambiguation_rules(VERB_DISAMBIG_TXT)

    lines = [
        "# Closed vocabularies (reference)",
        "",
        "This file summarizes the **taxonomy** used by DeepASMR-NSpeech and SVO-AQA.",
        "Full CSV tables, alias maps, LLM prompts, and evaluation scripts are in the",
        "**GitHub code repository** (not duplicated here).",
        "",
        "---",
        "",
        "## Verb taxonomy · 37 verbs → 18 superclasses",
        "",
    ]

    single, multi = [], []
    for row in verb_rows:
        cls = row["class"]
        verbs = [v.strip() for v in (row.get("verbs") or "").split(";") if v.strip()]
        hint = (row.get("class_hint") or "").strip()
        entry = (cls, verbs, hint)
        if len(verbs) <= 1:
            single.append(entry)
        else:
            multi.append(entry)

    lines.append("### Single-verb superclasses (1 verb each)")
    lines.append("")
    for cls, verbs, hint in single:
        lines.append(f"- **{cls}** → `{verbs[0]}` — {hint}")
    lines.append("")
    lines.append("### Multi-verb superclasses (within-class disambiguation applies)")
    lines.append("")
    for cls, verbs, hint in multi:
        verb_str = ", ".join(f"`{v}`" for v in verbs)
        lines.append(f"#### `{cls}`")
        lines.append(f"- **Verbs:** {verb_str}")
        lines.append(f"- **Class cue:** {hint}")
        title, rules = disambig.get(cls, ("", []))
        if title:
            lines.append(f"- **Disambiguation** ({title}):")
            for rule in rules:
                lines.append(f"  - {rule}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Material taxonomy · 12 superclasses",
        "",
        "Used in SVO-AQA material multiple-choice questions (`object_material_mc`,",
        "`subject_material_mc`). Clip-level material labels are **not** shipped in",
        "train/test JSON; MC options carry the material gold labels.",
        "",
        "Note: the release name is **`clay`** (modeling pack / wet sticky sound),",
        "not `ceramic`.",
        "",
    ])
    for row in mat_rows:
        cls = row["class"]
        hint = (row.get("hint") or row.get("class_hint") or "").strip()
        lines.append(f"- **{cls}** — {hint}")
    lines.extend([
        "",
        "---",
        "",
        "## GitHub (full artifacts)",
        "",
        "For reproducibility, obtain from the project repository:",
        "",
        "- `vocab/asmr_verbs_en.csv` — 37 verb short hints",
        "- `vocab/asmr_verb_classes_en.csv` — verb → superclass",
        "- `vocab/asmr_material_classes_compact_en.csv` — 12 material classes",
        "- `vocab/stage2_class_disambiguation.txt` — fine-grained rules",
        "- `vocab/verb_alias.csv`, `vocab/material_alias.csv` — alias maps",
        "- Annotation prompts under `annotation/`",
        "",
    ])

    VOCAB_REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_REF_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return VOCAB_REF_PATH


def write_release_manifest(stats: dict) -> None:
    manifest = {
        "dataset": "DeepASMR-NSpeech",
        "prepared_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **stats,
    }
    path = OUT_ROOT / "release_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataset_card() -> Path:
    """Install the Hugging Face card as the staged dataset README."""
    if not DATASET_CARD_SRC.is_file():
        raise FileNotFoundError(DATASET_CARD_SRC)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "README.md"
    shutil.copy2(DATASET_CARD_SRC, path)
    return path


def export_audio_dependent_subset(bank_path: Path, out_path: Path) -> dict:
    """Copy the frozen paper subset after validating it against the public bank."""
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    subset = json.loads(SVO_AQA_SUBSET_SRC.read_text(encoding="utf-8"))
    bank_ids = {str(row["question_id"]) for row in bank.get("questions") or []}
    subset_ids = {str(row["question_id"]) for row in subset.get("questions") or []}
    missing = subset_ids - bank_ids
    if missing:
        raise ValueError(f"audio-dependent subset has {len(missing)} IDs absent from bank")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SVO_AQA_SUBSET_SRC, out_path)
    return {
        "file": out_path.relative_to(OUT_ROOT).as_posix(),
        "n_questions": len(subset_ids),
        "n_clips": len({row.get("audio_id") for row in subset.get("questions") or []}),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare DeepASMR-NSpeech for Hugging Face")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--labels-only", action="store_true", help="Only rewrite JSON + benchmark; no audio")
    mode.add_argument("--link-audio", action="store_true", help="Symlink audio files (default)")
    mode.add_argument("--copy-audio", action="store_true", help="Copy audio files (for upload bundle)")
    p.add_argument(
        "--split",
        choices=("all", "train", "test", "svo-aqa"),
        default="all",
        help="Which split(s) to materialize",
    )
    p.add_argument("--skip-svo-aqa", action="store_true", help="Do not copy SVO-AQA/test_subset (431 MC)")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--train-label", type=Path, default=TRAIN_LABEL_SRC)
    p.add_argument("--test-label", type=Path, default=TEST_LABEL_SRC)
    p.add_argument("--train-audio-root", type=Path, default=TRAIN_AUDIO_PREFIX)
    p.add_argument("--svo-aqa-source", type=Path, default=MC_SAME_TRUE_SRC)
    return p.parse_args()


def main() -> int:
    global OUT_ROOT, TRAIN_LABEL_SRC, TEST_LABEL_SRC, TRAIN_AUDIO_PREFIX
    global MC_SAME_TRUE_SRC, CREATOR_MAP_PATH, VOCAB_REF_PATH
    args = parse_args()
    OUT_ROOT = args.out_root
    TRAIN_LABEL_SRC = args.train_label
    TEST_LABEL_SRC = args.test_label
    TRAIN_AUDIO_PREFIX = args.train_audio_root
    MC_SAME_TRUE_SRC = args.svo_aqa_source
    CREATOR_MAP_PATH = OUT_ROOT.parent / "build" / "private" / "creator_map.json"
    VOCAB_REF_PATH = OUT_ROOT / "vocab" / "closed_vocab_reference.md"
    copy_audio = bool(args.copy_audio)
    link_audio = bool(args.link_audio) or (not args.labels_only and not args.copy_audio)

    stats: Dict[str, Any] = {}
    stats["dataset_card"] = write_dataset_card().relative_to(OUT_ROOT).as_posix()

    creator_map: Dict[str, str] = {}
    if args.split in ("all", "train", "test"):
        label_paths = []
        if args.split in ("all", "train"):
            label_paths.append(TRAIN_LABEL_SRC)
        if args.split in ("all", "test"):
            label_paths.append(TEST_LABEL_SRC)
        creator_map = build_creator_id_map(*label_paths)
        write_creator_map(creator_map)
        stats["creator_map"] = {"n_creators": len(creator_map)}

    if args.split in ("all", "train"):
        train_stats = export_label_json(
            TRAIN_LABEL_SRC,
            OUT_ROOT / "train" / "train_label.json",
            train_rel_wav,
            creator_map,
        )
        stats["train_label"] = train_stats
        if link_audio or copy_audio:
            _, rows = load_label_rows(TRAIN_LABEL_SRC)
            stats["train_audio"] = materialize_train_audio(rows, copy=copy_audio)

    if args.split in ("all", "test"):
        test_stats = export_label_json(
            TEST_LABEL_SRC,
            OUT_ROOT / "test" / "test_label.json",
            test_rel_wav,
            creator_map,
        )
        stats["test_label"] = test_stats
        if link_audio or copy_audio:
            _, rows = load_label_rows(TEST_LABEL_SRC)
            stats["test_audio"] = materialize_test_audio(rows, copy=copy_audio)

    if args.split in ("all", "svo-aqa") and not args.skip_svo_aqa:
        out_dir = OUT_ROOT / SVO_AQA_REL
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        stats["svo_aqa"] = export_svo_aqa_bank(
            MC_SAME_TRUE_SRC,
            out_dir / "bank.json",
        )
        stats["svo_aqa"]["audio_dependent_subset"] = export_audio_dependent_subset(
            out_dir / "bank.json",
            out_dir / "audio_dependent_subset.v1.json",
        )

    if args.split in ("all", "train", "test", "svo-aqa"):
        vocab_path = generate_vocab_reference()
        stats["vocab_reference"] = vocab_path.relative_to(OUT_ROOT).as_posix()

    write_release_manifest(stats)
    mode = "labels-only" if args.labels_only else ("copy" if copy_audio else "symlink")
    print(f"[DONE] DeepASMR-NSpeech prepared at {OUT_ROOT} (audio mode: {mode})")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
