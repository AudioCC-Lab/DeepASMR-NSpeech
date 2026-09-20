#!/usr/bin/env python3
"""Normalize draft SVO labels to the released vocabulary.

用法:
  python annotation/test/run_svo_normalize.py --dry-run
  python annotation/test/run_svo_normalize.py --input annotation/test/outputs/seg_label_svo.json
  python annotation/test/run_svo_normalize.py --continue-on-error
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
LIB_DIR = ROOT / "lib"
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
OUT_DIR = ROOT / "outputs"

for p in (LIB_DIR, ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from env_file import load_env_early  # noqa: E402
from events_io import event_key, seg_label  # noqa: E402
from annotation.common.info_common import (  # noqa: E402
    DEFAULT_SILICONFLOW_BASE_URL,
    TokenUsageAccumulator,
    atomic_write_json,
    call_llm_json_with_usage,
    clamp_confidence,
    estimate_cost_cny,
    fill_template,
    load_info_full_payload,
    load_project_env,
    load_verb_alias_map,
    load_verb_class_lexicon,
    make_openai_client,
    norm_noun,
    norm_token,
    resolve_verb_canonical,
    resolve_verb_lexicon,
)
from annotation.common.prompt_util import load_marked_prompt  # noqa: E402
from deepasmr_nspeech import RunSummary  # noqa: E402

DEFAULT_INPUT = OUT_DIR / "seg_label_svo_qwen3635ba3b.json"
DEFAULT_NOUN_PROMPT = PROMPTS_DIR / "svo_normalize_noun_llm.txt"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
MODEL_PRICING_CNY = {"Qwen/Qwen3.6-35B-A3B": {"input": 0.40, "output": 3.20}}

_COLOR_RE = re.compile(
    r"\b(red|blue|green|pink|purple|black|white|yellow|orange|teal|amber|golden|silver|metallic|glittery)\b",
    re.I,
)
_LIQUID_RE = re.compile(
    r"\b(water|liquid|soapy|submerged|effervesc|bath bomb|fluid|fizz|"
    r"bubbles?\b|contains liquid|agitated by airflow)\b",
    re.I,
)
_EAR_MODEL_CTX = re.compile(
    r"\b(silicone|binaural|3dio|mannequin|dummy head|ear model|ear replica|ear-shaped)\b",
    re.I,
)


def _is_fail(val: Any) -> bool:
    return str(val or "").strip().lower() == "fail"


def _is_filled(val: Any) -> bool:
    return val is not None and not _is_fail(val) and str(val).strip() != ""


def build_caption(subject: Any, verb: Any, object_: Any) -> Optional[str]:
    parts: List[str] = []
    for val in (subject, verb, object_):
        if _is_filled(val):
            parts.append(str(val).strip())
    return " ".join(parts) if parts else None


def strip_color_words(phrase: str) -> str:
    t = _COLOR_RE.sub("", phrase)
    return re.sub(r"\s+", " ", t).strip()


def _tool_material(phrase: str) -> str:
    t = phrase.lower()
    if any(k in t for k in ("bamboo", "wooden", "wood")):
        return "wooden tool"
    if any(k in t for k in ("metal", "metallic", "spring", "steel")):
        return "metal tool"
    if "plastic" in t:
        return "plastic tool"
    if any(k in t for k in ("cotton", "fluffy", "swab")):
        return "cotton tool"
    if any(k in t for k in ("silicone", "rubber")):
        return "silicone tool"
    return "metal tool"


def normalize_tool_phrase(phrase: str) -> str:
    t = phrase.lower()
    if any(k in t for k in ("ear pick", "earpick", "ear picks", "earpicks", "ear stick", "ear cleaner", "ear cleaning")):
        return _tool_material(t)
    if "tweezer" in t:
        return "metal tool"
    if "spoolie" in t:
        return "plastic tool"
    if "cotton swab" in t or "cotton tip" in t or "cotton tips" in t:
        return "cotton swab"
    if t.endswith(" brush") or t.endswith(" brushes") or "paint brush" in t or "makeup brush" in t or "hairbrush" in t:
        if "wooden" in t or "wood" in t:
            return "wooden brush"
        return "brush"
    if "fork" in t and "brush" in t:
        return "brush"
    return phrase


def normalize_ear_receiver(phrase: str, row: dict) -> str:
    t = phrase.lower()
    ctx = " ".join(
        str(row.get(k) or "")
        for k in ("seg_label", "visual_description", "visual_event", "title")
    )
    if t in ("ear", "ears", "left ear", "right ear", "ear canal", "eardrum"):
        if _EAR_MODEL_CTX.search(ctx) or "ear cleaning" in ctx.lower() or "silicone" in ctx.lower():
            return "silicone ear" if t != "ears" else "silicone ears"
    if t in ("earpod", "earpods", "earphone", "earphones", "earbud", "earbuds", "earphone mic"):
        if _EAR_MODEL_CTX.search(ctx):
            return "silicone ear"
        return "mic"
    if "crystal ear" in t:
        return "glass ear"
    if "wooden ear" in t:
        return "wooden ear"
    if "dummy head" in t or "wooden head" in t:
        return "silicone ear"
    return phrase


def normalize_noun_phrase(raw: Any, row: dict, *, field: str) -> Optional[str]:
    if raw is None:
        return None
    if _is_fail(raw):
        return "fail"
    t = norm_noun(raw)
    if not t:
        return None
    t = strip_color_words(t)
    t = normalize_tool_phrase(t)
    t = normalize_ear_receiver(t, row)
    if "sealing wax" in t:
        t = t.replace("sealing wax", "wax").strip()
    if t == "sealing wax sticker" or t == "sealing wax stickers":
        t = "wax stickers"
    t = norm_noun(t) or t
    if t in ("microphone", "microphones", "mics"):
        t = "mic"
    if "recorder" in t and "mic" not in t:
        t = "mic"
    return t or None


def has_liquid_context(row: dict) -> bool:
    blob = " ".join(
        str(row.get(k) or "")
        for k in ("seg_label", "visual_description", "visual_event", "title")
    )
    return bool(_LIQUID_RE.search(blob))


def infer_bubble_object(row: dict) -> str:
    blob = " ".join(
        str(row.get(k) or "")
        for k in ("seg_label", "visual_description", "visual_event")
    ).lower()
    if "soapy" in blob or "soap" in blob:
        return "soapy water"
    if "bath bomb" in blob:
        return "water"
    if "sphere" in blob or "orb" in blob:
        return "plastic spheres"
    return "water"


def apply_blow_bubble_rule(
    row: dict,
    verb: str,
    subject: Any,
    obj: Any,
    *,
    legacy_verb: str = "",
) -> Tuple[str, Any, Any, Optional[str]]:
    """吹气 vs 冒泡：仅对吹气入液体场景改为 air bubbling water。"""
    leg = (legacy_verb or row.get("verb") or "").lower()
    sl = (row.get("seg_label") or "").lower()
    is_blow_case = leg in ("blowing", "blow") or "blow" in sl or str(subject or "").lower() in ("air", "straw", "mouth", "lips")

    if verb == "blowing" and has_liquid_context(row):
        bubble_obj = infer_bubble_object(row)
        return "bubbling", "air", bubble_obj, "blow_liquid_to_bubbling"

    if verb in ("fizz", "fizzing") and has_liquid_context(row):
        bubble_obj = infer_bubble_object(row)
        subj = "air" if is_blow_case else (subject or "air")
        return "bubbling", subj, obj or bubble_obj, "fizz_to_bubbling"

    return verb, subject, obj, None


def resolve_cleaning_verb(row: dict, subject: Any) -> Optional[str]:
    sub = (str(subject or "")).lower()
    obj = (str(row.get("object") or "")).lower()
    vis = (str(row.get("visual_description") or "")).lower()
    sl = (str(row.get("seg_label") or "")).lower()
    ctx = obj + vis + sl + sub
    if not any(k in ctx for k in ("ear", "mic", "silicone")):
        return None
    if any(k in sub for k in ("brush", "spoolie", "bristle", "makeup", "fluffy ball")):
        return "brushing"
    if any(k in sub for k in ("cotton", "swab", "pad", "wipe", "cloth", "konjac", "sponge")):
        return "wiping"
    if any(k in sub for k in ("finger", "nail", "hand")) and "pick" not in sub and "stick" not in sub:
        return "rubbing"
    if any(
        k in sub + vis
        for k in (
            "pick", "stick", "bamboo", "metal", "wooden", "tweezer", "tool", "rod",
            "spatula", "earpick", "ear pick",
        )
    ):
        return "scraping"
    return "scraping"


def fix_subject_eq_object(row: dict, subject: Any, verb: str, obj: Any) -> Tuple[Any, Any, Optional[str]]:
    """规则修正 subject==object。"""
    if not _is_filled(subject) or not _is_filled(obj) or subject != obj:
        return subject, obj, None
    sub, o = str(subject), str(obj)
    sl = (row.get("seg_label") or "").lower()
    vis = (row.get("visual_description") or "").lower()

    if sub == "hair" and verb == "brushing":
        if "brush" in vis or "bristle" in vis or "hairbrush" in vis:
            return "brush", "hair", "hair_brush_subject"
        return "brush", "hair", "hair_brush_subject"

    if sub in ("nails", "fingernails") and "click" in sl:
        if "mic" in vis or "microphone" in vis:
            return "nails", "mic", "clicking_mic"
        return "nails", "hard surface", "clicking_surface"

    if sub == "soap" and verb in ("crushing", "cutting", "carving", "slicing"):
        if "hand" in vis:
            return "hands", "soap", "crushing_soap"
        return "metal tool", "soap", "carving_soap"

    if sub == "mic" and verb == "rubbing":
        return "hands", "mic", "mic_rubbing_hands"

    if "modeling pack" in sl and sub == o:
        return "modeling pack", "silicone ear", "modeling_pack_subject"

    return subject, obj, None


def resolve_melting_verb(row: dict) -> Optional[str]:
    sl = (row.get("seg_label") or "").lower()
    vis = (row.get("visual_description") or "").lower()
    if "bath bomb" in sl or "bath bomb" in vis:
        if any(k in vis for k in ("leaf", "leaves", "rustl")):
            return "pressing"
        if any(k in vis for k in ("fizz", "bubble", "dissolv", "effervesc", "foam")):
            return "bubbling"
    if any(k in vis for k in ("fizz", "bubble", "dissolv", "effervesc")):
        return "bubbling"
    return "crushing"


def resolve_verb_rule(
    raw: str,
    row: dict,
    subject: Any,
    lex: Any,
    alias_map: Dict[str, str],
) -> Tuple[Optional[str], str]:
    if not raw or _is_fail(raw):
        return None, "missing"
    if raw == "cleaning":
        cv = resolve_cleaning_verb(row, subject)
        if cv and cv in lex.all_verbs:
            return cv, "rule_cleaning"
    if raw == "melting":
        mv = resolve_melting_verb(row)
        if mv in lex.all_verbs:
            return mv, "rule_melting"
    canon = resolve_verb_lexicon(raw, lex)
    if canon:
        return canon, "lexicon"
    canon = resolve_verb_canonical(raw, lex, alias_map)
    if canon:
        return canon, "alias"
    return None, "unmapped"


def resolve_verb_visual_fallback(row: dict, raw_verb: str) -> Optional[str]:
    """LLM 失败时的视觉/标签兜底。"""
    sl = (row.get("seg_label") or "").lower()
    vis = (row.get("visual_description") or "").lower()
    if _is_fail(raw_verb):
        if "kissing" in sl or "kiss" in vis:
            return "licking"
        if "popping candy" in sl or "popping candy" in vis:
            return "popping"
        if "crunch" in vis or "stretch" in vis:
            if "slime" in sl:
                return "crunching"
        if "rub" in vis or "press" in vis or "tap" in vis:
            return "rubbing"
    return None


def normalize_prop_subject(subject: Any, row: dict) -> Any:
    if not _is_filled(subject):
        return subject
    t = str(subject).lower()
    if "unicorn" in t:
        return "plastic toy"
    return subject


def needs_verb_llm(row: dict, verb: str, verb_source: str) -> bool:
    if _is_fail(verb) or row.get("verb_failed"):
        return True
    return verb_source in ("missing", "unmapped")


def needs_noun_llm(row: dict, subject: Any, obj: Any) -> bool:
    if _is_fail(subject) or row.get("subject_failed"):
        return True
    if _is_fail(obj) or row.get("object_failed"):
        return True
    if _is_filled(subject) and _is_filled(obj) and subject == obj:
        return True
    return False


def run_verb_llm(
    *,
    row: dict,
    subject: Any,
    obj: Any,
    legacy_verb: str,
    lex: Any,
    alias_map: Dict[str, str],
    client: Any,
    model: str,
    system_prompt: str,
    user_tpl: str,
    usage_acc: TokenUsageAccumulator,
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    user_text = fill_template(
        user_tpl,
        {
            "SEG_LABEL": seg_label(row) or "(none)",
            "VISUAL_DESCRIPTION": (row.get("visual_description") or "").strip() or "(none)",
            "VISUAL_EVENT": (row.get("visual_event") or "").strip() or "(none)",
            "SUBJECT": str(subject or "null"),
            "OBJECT": str(obj or "null"),
            "LEGACY_VERB": legacy_verb or "fail",
        },
    )
    parsed = call_llm_json_with_usage(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        usage_acc=usage_acc,
        stage="svo_norm_verb_llm",
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        max_retries=int(args.max_retries),
        sleep_base=float(args.sleep_base),
    )
    verb_raw = str(parsed.get("verb") or "").strip()
    verb_out = resolve_verb_canonical(verb_raw, lex, alias_map)
    if verb_out not in lex.all_verbs:
        return None, f"LLM verb OOB: {verb_raw!r}", None
    reason = (parsed.get("norm_reason") or parsed.get("llm_verify_reason") or "").strip()[:500]
    return verb_out, None, reason


def run_noun_llm(
    *,
    row: dict,
    verb: str,
    subject: Any,
    obj: Any,
    client: Any,
    model: str,
    system_prompt: str,
    user_tpl: str,
    usage_acc: TokenUsageAccumulator,
    args: argparse.Namespace,
) -> Tuple[Any, Any, Optional[str], Optional[str]]:
    user_text = fill_template(
        user_tpl,
        {
            "SEG_LABEL": seg_label(row) or "(none)",
            "VISUAL_DESCRIPTION": (row.get("visual_description") or "").strip() or "(none)",
            "LEGACY_SUBJECT": str(subject or "fail"),
            "LEGACY_OBJECT": str(obj or "null"),
        },
    )
    parsed = call_llm_json_with_usage(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        usage_acc=usage_acc,
        stage="svo_norm_noun_llm",
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        max_retries=int(args.max_retries),
        sleep_base=float(args.sleep_base),
    )
    sub_raw = parsed.get("subject")
    obj_raw = parsed.get("object")
    sub = None if sub_raw is None or str(sub_raw).strip().lower() in ("null", "none", "") else norm_noun(sub_raw)
    obj = None
    if obj_raw is not None and str(obj_raw).strip().lower() not in ("null", "none", ""):
        obj = norm_noun(obj_raw)
    if sub:
        sub = normalize_noun_phrase(sub, row, field="subject")
    if obj:
        obj = normalize_noun_phrase(obj, row, field="object")
    reason = (parsed.get("llm_verify_reason") or "").strip()[:500]
    return sub, obj, reason, None


def normalize_row(
    row: dict,
    *,
    lex: Any,
    alias_map: Dict[str, str],
    client: Any,
    model: str,
    verb_sys: str,
    verb_user: str,
    noun_sys: str,
    noun_user: str,
    usage_acc: TokenUsageAccumulator,
    args: argparse.Namespace,
    use_llm: bool,
) -> dict:
    out = copy.deepcopy(row)
    legacy = {
        "subject": row.get("subject"),
        "verb": row.get("verb"),
        "object": row.get("object"),
        "caption": row.get("caption"),
    }
    out["legacy_svo"] = legacy

    subject = normalize_noun_phrase(row.get("subject"), row, field="subject")
    subject = normalize_prop_subject(subject, row)
    obj = normalize_noun_phrase(row.get("object"), row, field="object")
    raw_verb = str(row.get("verb") or "").strip()
    verb = norm_token(raw_verb) if raw_verb and not _is_fail(raw_verb) else raw_verb

    verb, verb_source = resolve_verb_rule(raw_verb, row, subject, lex, alias_map)
    norm_notes: List[str] = []

    if verb:
        verb, subject, obj, note = apply_blow_bubble_rule(row, verb, subject, obj, legacy_verb=raw_verb)
        if note:
            verb_source = note
            norm_notes.append(note)

    if verb == "bubbling" and "bath bomb" in (row.get("seg_label") or "").lower():
        if not _is_filled(subject) or str(subject).lower() == "air":
            subject = "bath bomb"
        if not obj and has_liquid_context(row):
            obj = infer_bubble_object(row)
            norm_notes.append("bath_bomb_object_water")

    subject, obj, eq_note = fix_subject_eq_object(row, subject, verb or raw_verb, obj)
    if eq_note:
        norm_notes.append(eq_note)

    noun_source = "rule"
    noun_reason: Optional[str] = None
    verb_reason: Optional[str] = None

    if use_llm and needs_verb_llm(row, verb or raw_verb, verb_source or "missing"):
        # 视觉与标签不一致时优先视觉（如 bath bomb 标签但画面是搓叶子）
        vis = (row.get("visual_description") or "").lower()
        if _is_fail(raw_verb) and any(k in vis for k in ("rustl", "leaf", "leaves")):
            verb = "pressing"
            verb_source = "rule_visual_rustle"
            if _is_fail(obj):
                obj = "leaves"
        elif _is_fail(raw_verb) and "peel" in vis:
            verb = "tearing"
            verb_source = "rule_visual_peel"
        else:
            verb_new, err, vreason = run_verb_llm(
                row=row, subject=subject, obj=obj, legacy_verb=raw_verb,
                lex=lex, alias_map=alias_map, client=client, model=model,
                system_prompt=verb_sys, user_tpl=verb_user,
                usage_acc=usage_acc, args=args,
            )
            if err:
                fb = resolve_verb_visual_fallback(row, raw_verb)
                if fb:
                    verb = fb
                    verb_source = "rule_visual_fallback"
                    norm_notes.append("verb_llm_fallback")
                else:
                    out["norm_error"] = err
            elif verb_new:
                verb = verb_new
                verb_source = "llm"
                verb_reason = vreason
                v, subject, obj, note = apply_blow_bubble_rule(
                    row, verb, subject, obj, legacy_verb=raw_verb
                )
                if note:
                    verb_source = f"llm+{note}"
                    norm_notes.append(note)
                verb = v

    if use_llm and needs_noun_llm(row, subject, obj):
        sub_new, obj_new, nreason, nerr = run_noun_llm(
            row=row, verb=verb or "", subject=subject, obj=obj,
            client=client, model=model,
            system_prompt=noun_sys, user_tpl=noun_user,
            usage_acc=usage_acc, args=args,
        )
        if nerr:
            out["norm_error"] = (out.get("norm_error") or "") + " " + nerr
        else:
            if sub_new:
                subject = sub_new
            if obj_new is not None or _is_fail(row.get("object")):
                obj = obj_new
            noun_source = "llm"
            noun_reason = nreason

    subject, obj, eq_note2 = fix_subject_eq_object(row, subject, verb or raw_verb, obj)
    if eq_note2:
        norm_notes.append(eq_note2)

    out["subject"] = subject if _is_filled(subject) else (None if subject is None else subject)
    out["verb"] = verb
    out["object"] = obj if obj is not None else None
    out["caption"] = build_caption(out["subject"], out["verb"], out["object"])
    out["subject_failed"] = _is_fail(out["subject"])
    out["verb_failed"] = _is_fail(out["verb"]) or not out["verb"]
    out["object_failed"] = _is_fail(out["object"])
    out["verb_source"] = verb_source
    out["noun_source"] = noun_source
    out["norm_notes"] = norm_notes
    if verb_reason:
        out["verb_norm_reason"] = verb_reason
    if noun_reason:
        out["noun_norm_reason"] = noun_reason
    out["needs_manual_review"] = bool(
        out["subject_failed"] or out["verb_failed"] or out["object_failed"] or out.get("norm_error")
    )
    if out["needs_manual_review"] and not out.get("manual_review_reason"):
        reasons = []
        if out["subject_failed"]:
            reasons.append("subject_fail")
        if out["verb_failed"]:
            reasons.append("verb_fail")
        if out["object_failed"]:
            reasons.append("object_fail")
        if out.get("norm_error"):
            reasons.append("norm_error")
        out["manual_review_reason"] = ";".join(reasons)
    return out


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_normalized"):
        return input_path
    return input_path.with_name(f"{stem}_normalized.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVO 标注标准化")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verb-alias-csv", type=Path, default=REPO_ROOT / "vocab" / "verb_alias.csv")
    p.add_argument("--verb-prompt", type=Path, default=PROMPTS_DIR / "svo_normalize_verb_llm.txt")
    p.add_argument("--noun-prompt", type=Path, default=DEFAULT_NOUN_PROMPT)
    p.add_argument("--model", default=os.environ.get("INFO_FULL_VERIFY_TEXT_MODEL") or DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stems", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true", help="仅统计，不调 LLM、不写文件")
    p.add_argument("--no-llm", action="store_true", help="仅规则+alias，不调 LLM")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep-base", type=float, default=1.5)
    return p.parse_args()


def print_dry_run_stats(rows: List[dict], lex: Any, alias_map: Dict[str, str]) -> None:
    verb_src = Counter()
    need_v = need_n = 0
    same = 0
    blow_fix = 0
    for row in rows:
        sub = normalize_noun_phrase(row.get("subject"), row, field="subject")
        obj = normalize_noun_phrase(row.get("object"), row, field="object")
        raw_v = str(row.get("verb") or "")
        v, src = resolve_verb_rule(raw_v, row, sub, lex, alias_map)
        if v:
            v2, _, _, note = apply_blow_bubble_rule(row, v, sub, obj, legacy_verb=raw_v)
            if note:
                blow_fix += 1
            verb_src[src] += 1
        else:
            verb_src["unmapped"] += 1
        if needs_verb_llm(row, v or raw_v, src or "missing"):
            need_v += 1
        if needs_noun_llm(row, sub, obj):
            need_n += 1
        if _is_filled(sub) and _is_filled(obj) and sub == obj:
            same += 1
    print(f"rows={len(rows)} verb_sources={dict(verb_src)}")
    print(f"need_verb_llm={need_v} need_noun_llm={need_n} subject_eq_object={same} blow_bubble_fix={blow_fix}")


def main() -> int:
    args = parse_args()
    load_env_early(sys.argv, [ROOT.parent / ".env", ROOT / ".env"])
    load_project_env()

    in_path = args.input.resolve()
    out_path = (args.output or default_output_path(in_path)).resolve()
    wrapper, rows = load_info_full_payload(in_path)

    lex = load_verb_class_lexicon(REPO_ROOT / "vocab" / "asmr_verb_classes_en.csv")
    alias_map = load_verb_alias_map(args.verb_alias_csv.resolve())

    if args.stems:
        stems = set(args.stems)
        rows = [r for r in rows if event_key(r) in stems]
    if args.limit:
        rows = rows[: args.limit]

    if args.dry_run:
        print_dry_run_stats(rows, lex, alias_map)
        return 0

    use_llm = not args.no_llm
    client = None
    verb_sys = verb_user = noun_sys = noun_user = ""
    if use_llm:
        api_key = (os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            print("[ERROR] 需要 SILICONFLOW_API_KEY 或 OPENAI_API_KEY", file=sys.stderr)
            return 1
        verb_sys, verb_user = load_marked_prompt(args.verb_prompt.resolve())
        noun_sys, noun_user = load_marked_prompt(args.noun_prompt.resolve())
        client = make_openai_client(
            api_key, os.environ.get("OPENAI_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL)
        )

    usage_acc = TokenUsageAccumulator()
    meta = dict(wrapper.get("meta") or {})
    meta["normalize_pipeline"] = "test_v5_svo_normalize"
    meta["normalize_input"] = str(in_path)
    meta["normalize_model"] = args.model if use_llm else None
    meta["normalize_created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_rows: List[dict] = []
    errors = 0
    run_summary = RunSummary("annotation.test.svo_normalize")
    t0 = time.time()
    for i, row in enumerate(rows):
        try:
            norm = normalize_row(
                row,
                lex=lex,
                alias_map=alias_map,
                client=client,
                model=args.model,
                verb_sys=verb_sys,
                verb_user=verb_user,
                noun_sys=noun_sys,
                noun_user=noun_user,
                usage_acc=usage_acc,
                args=args,
                use_llm=use_llm,
            )
            out_rows.append(norm)
            run_summary.add("success", item_id=event_key(row))
        except Exception as e:
            errors += 1
            bad = copy.deepcopy(row)
            bad["norm_error"] = str(e)
            bad["needs_manual_review"] = True
            out_rows.append(bad)
            run_summary.add("failed", item_id=event_key(row), detail=e)
            if not args.continue_on_error:
                raise
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(rows)}] elapsed={time.time()-t0:.0f}s usage={usage_acc.to_dict()}")

    meta["normalize_usage"] = usage_acc.to_dict()
    rates = MODEL_PRICING_CNY.get(args.model, {"input": 0.40, "output": 3.20})
    meta["normalize_cost_cny"] = estimate_cost_cny(
        usage_acc, input_per_m=rates["input"], output_per_m=rates["output"]
    )
    meta["normalize_n_ok"] = sum(1 for r in out_rows if not r.get("needs_manual_review"))
    meta["normalize_n_review"] = sum(1 for r in out_rows if r.get("needs_manual_review"))
    meta["normalize_n_error"] = errors

    payload = {**wrapper, "meta": meta, "data": out_rows}
    atomic_write_json(out_path, payload)

    vsrc = Counter(r.get("verb_source") for r in out_rows)
    nsrc = Counter(r.get("noun_source") for r in out_rows)
    print(f"Wrote {out_path}")
    print(f"verb_source: {dict(vsrc)}")
    print(f"noun_source: {dict(nsrc)}")
    print(f"needs_manual_review: {meta['normalize_n_review']}/{len(out_rows)}")
    print(f"usage: {usage_acc.to_dict()} cost_cny={meta['normalize_cost_cny']}")
    run_summary.finish(
        out_path.with_suffix(".run_summary.json"),
        metadata={
            "needs_manual_review": meta["normalize_n_review"],
            "api_calls": usage_acc.api_calls,
            "tokens": usage_acc.total_tokens,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
