#!/usr/bin/env python3
"""Use one multimodal LLM call to draft SVO annotations.

流程：视觉判断事件 → 仅从 seg_label/title 选词填 SVO → 填不出则 fail。

用法:
  python annotation/test/run_seg_label_svo.py --dry-run
  python annotation/test/run_seg_label_svo.py --continue-on-error
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
LIB_DIR = ROOT / "lib"
PROMPTS_DIR = ROOT / "prompts"
OUT_DIR = ROOT / "outputs"

for p in (LIB_DIR, ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from env_file import load_env_early  # noqa: E402
from events_io import event_key, load_test_events, seg_label  # noqa: E402
from annotation.common.info_common import (  # noqa: E402
    DEFAULT_SILICONFLOW_BASE_URL,
    TokenUsageAccumulator,
    atomic_write_json,
    call_llm_json_with_usage,
    clamp_confidence,
    estimate_cost_cny,
    fill_template,
    load_project_env,
    make_openai_client,
    norm_noun,
    norm_token,
)
from annotation.common.prompt_util import load_marked_prompt  # noqa: E402
from deepasmr_nspeech import RunSummary  # noqa: E402

DEFAULT_INPUT = ROOT / "test_v5.json"
DEFAULT_OUT = OUT_DIR / "seg_label_svo.json"
DEFAULT_PROMPT = PROMPTS_DIR / "seg_label_svo_extract.txt"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
MODEL_PRICING_CNY = {"Qwen/Qwen3.6-35B-A3B": {"input": 0.40, "output": 3.20}}


def model_slug(model: str) -> str:
    s = model.split("/")[-1].lower().replace("-", "").replace(".", "")
    return s[:24] or "model"


def _row_done(row: Optional[dict]) -> bool:
    return bool(row and row.get("status") == "ok" and not row.get("error"))


def _is_fail(val: Any) -> bool:
    return str(val or "").strip().lower() == "fail"


def _is_filled(val: Any) -> bool:
    return val is not None and not _is_fail(val) and str(val).strip() != ""


def _parse_svo_field(raw: Any, *, as_noun: bool) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "null":
        return None
    if s.lower() == "fail":
        return "fail"
    return norm_noun(s) if as_noun else (norm_token(s) or None)


def build_caption(subject: Any, verb: Any, object_: Any) -> Optional[str]:
    parts: List[str] = []
    for val in (subject, verb, object_):
        if _is_filled(val):
            parts.append(str(val).strip())
    return " ".join(parts) if parts else None


def run_extract_llm(
    *,
    event: dict,
    client: Any,
    model: str,
    system_prompt: str,
    user_tpl: str,
    usage_acc: TokenUsageAccumulator,
    args: argparse.Namespace,
) -> Tuple[Optional[dict], Optional[str]]:
    user_text = fill_template(
        user_tpl,
        {
            "SEG_LABEL": seg_label(event) or "(none)",
            "TITLE": (event.get("title") or "").strip() or "(none)",
            "VISUAL_DESCRIPTION": (event.get("visual_description") or "").strip() or "(none)",
        },
    )
    try:
        parsed = call_llm_json_with_usage(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_text=user_text,
            usage_acc=usage_acc,
            stage="seg_label_svo_extract",
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            max_retries=int(args.max_retries),
            sleep_base=float(args.sleep_base),
        )
    except Exception as e:
        return None, str(e)
    return parsed, None


def process_event(
    *,
    event: dict,
    client: Any,
    model: str,
    system_prompt: str,
    user_tpl: str,
    usage_acc: TokenUsageAccumulator,
    args: argparse.Namespace,
) -> dict:
    wav = str(event.get("wav") or "").strip()
    out: Dict[str, Any] = {
        "wav": wav,
        "seg_label": seg_label(event),
        "title": event.get("title"),
        "author": event.get("author"),
        "visual_description": event.get("visual_description"),
    }

    parsed, err = run_extract_llm(
        event=event,
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_tpl=user_tpl,
        usage_acc=usage_acc,
        args=args,
    )
    if err or parsed is None:
        out.update(
            {
                "status": "fail",
                "error": err or "extract failed",
                "fail_reason": err or "extract failed",
                "needs_manual_review": True,
            }
        )
        return out

    subject = _parse_svo_field(parsed.get("subject"), as_noun=True)
    verb = _parse_svo_field(parsed.get("verb"), as_noun=False)
    object_ = _parse_svo_field(parsed.get("object"), as_noun=True)

    subject_failed = _is_fail(subject)
    verb_failed = _is_fail(verb)
    object_failed = _is_fail(object_)

    field_fail_reasons = parsed.get("field_fail_reasons")
    if not isinstance(field_fail_reasons, dict):
        field_fail_reasons = {}
    legacy_reason = str(parsed.get("fail_reason") or "").strip()
    if legacy_reason and verb_failed and "verb" not in field_fail_reasons:
        field_fail_reasons = {**field_fail_reasons, "verb": legacy_reason}

    has_any = _is_filled(subject) or _is_filled(verb) or _is_filled(object_)
    status = "ok" if has_any else "fail"

    caption = str(parsed.get("caption") or "").strip() or None
    if not caption and has_any:
        caption = build_caption(subject, verb, object_)

    needs_review = subject_failed or verb_failed or object_failed or not has_any

    meta_used = parsed.get("metadata_words_used")
    if not isinstance(meta_used, list):
        meta_used = []

    out.update(
        {
            "status": status,
            "visual_event": (parsed.get("visual_event") or "").strip()[:500],
            "subject": subject,
            "verb": verb,
            "object": object_,
            "caption": caption,
            "subject_failed": subject_failed,
            "verb_failed": verb_failed,
            "object_failed": object_failed,
            "field_fail_reasons": field_fail_reasons,
            "metadata_words_used": meta_used,
            "needs_manual_review": needs_review,
            "manual_review_reason": (
                "; ".join(f"{k}: {v}" for k, v in field_fail_reasons.items() if v)
                or (None if not needs_review else "all fields fail")
            ),
            "extract_reason": (parsed.get("llm_verify_reason") or "").strip()[:500],
            "extract_confidence": clamp_confidence(parsed.get("llm_verify_confidence", 5)),
        }
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="test_v5 单次 LLM：视觉定事件 + metadata 填 SVO")
    p.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument(
        "--model",
        default=os.environ.get("INFO_FULL_VERIFY_TEXT_MODEL")
        or os.environ.get("LABEL_PARSE_MODEL")
        or DEFAULT_MODEL,
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stems", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep-base", type=float, default=1.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_env_early(sys.argv, [ROOT.parent / ".env", ROOT / ".env"])
    load_project_env()

    model = (args.model or DEFAULT_MODEL).strip()
    out_path = args.out_json or (OUT_DIR / f"seg_label_svo_{model_slug(model)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.input_json.is_file():
        print(f"[ERROR] 输入不存在: {args.input_json}", file=sys.stderr)
        return 1

    wrapper, events = load_test_events(args.input_json)
    run_summary = RunSummary("annotation.test.svo_extract")

    existing: Dict[str, dict] = {}
    if args.resume and out_path.is_file():
        try:
            _, prev = load_test_events(out_path)
            existing = {str(r.get("wav") or ""): r for r in prev if r.get("wav")}
        except ValueError:
            existing = {}

    stems = set(args.stems) if args.stems else None
    pending: List[dict] = []
    for e in events:
        wav = str(e.get("wav") or "").strip()
        item_id = wav or event_key(e)
        if not wav:
            run_summary.add("skipped", item_id=item_id, reason="missing_audio_path")
            continue
        if not seg_label(e):
            run_summary.add("skipped", item_id=item_id, reason="missing_segment_label")
            continue
        if not (e.get("visual_description") or "").strip():
            run_summary.add("skipped", item_id=item_id, reason="missing_visual_description")
            continue
        if stems and event_key(e) not in stems:
            run_summary.add("skipped", item_id=item_id, reason="not_selected")
            continue
        if args.resume and _row_done(existing.get(wav)) and not args.overwrite:
            run_summary.add("skipped", item_id=item_id, reason="already_completed")
            continue
        pending.append(e)

    print(f"[INFO] pending={len(pending)} model={model} → {out_path}")

    if args.dry_run:
        print(f"[DRY-RUN] API calls ≈ {len(pending)} (1 per row)")
        return 0

    api_key = (os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("[ERROR] 需要 SILICONFLOW_API_KEY", file=sys.stderr)
        return 1

    system_prompt, user_tpl = load_marked_prompt(args.prompt)
    client = make_openai_client(api_key, os.environ.get("OPENAI_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL))
    usage_acc = TokenUsageAccumulator()

    merged: Dict[str, dict] = dict(existing)
    ok = fail = err = 0
    limit = max(0, args.limit)

    for idx, event in enumerate(pending):
        if limit > 0 and (ok + fail) >= limit:
            break
        wav = str(event.get("wav") or "")
        try:
            row = process_event(
                event=event,
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_tpl=user_tpl,
                usage_acc=usage_acc,
                args=args,
            )
        except Exception as e:
            row = {
                "wav": wav,
                "seg_label": seg_label(event),
                "status": "fail",
                "error": str(e),
                "fail_reason": str(e),
                "needs_manual_review": True,
            }
        merged[wav] = row
        if row.get("error") and row.get("status") != "ok":
            err += 1
            run_summary.add("failed", item_id=wav or event_key(event), detail=row.get("error"))
            if not args.continue_on_error:
                break
        elif row.get("status") == "ok":
            ok += 1
            run_summary.add("success", item_id=wav or event_key(event))
        else:
            fail += 1
            run_summary.add(
                "failed",
                item_id=wav or event_key(event),
                reason="annotation_failed",
                detail=row.get("fail_reason"),
            )

        name = Path(wav).name if wav else event_key(event)
        sf = row.get("subject_failed") or row.get("subject") == "fail"
        vf = row.get("verb_failed") or row.get("verb") == "fail"
        of_ = row.get("object_failed") or row.get("object") == "fail"
        print(
            f"[{idx+1}/{len(pending)}] {name} "
            f"s={row.get('subject')!r}{'!' if sf else ''} "
            f"v={row.get('verb')!r}{'!' if vf else ''} "
            f"o={row.get('object')!r}{'!' if of_ else ''} "
            f"caption={row.get('caption')!r}"
        )

        if (idx + 1) % max(1, args.checkpoint_every) == 0 or idx == len(pending) - 1:
            pricing = MODEL_PRICING_CNY.get(model, {})
            cost = (
                estimate_cost_cny(
                    usage_acc,
                    input_per_m=pricing.get("input", 0),
                    output_per_m=pricing.get("output", 0),
                )
                if pricing
                else {}
            )
            payload = {
                "meta": {
                    "pipeline": "test_v5_seg_label_svo_v2",
                    "model": model,
                    "input_json": str(args.input_json),
                    "prompt": str(args.prompt),
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "n_ok": ok,
                    "n_fail": fail,
                    "n_error": err,
                    "usage": usage_acc.to_dict(),
                    "cost_cny": cost,
                },
                "data": [merged[str(e.get("wav") or "")] for e in events if str(e.get("wav") or "") in merged]
                + [v for k, v in merged.items() if k not in {str(e.get("wav") or "") for e in events}],
            }
            atomic_write_json(out_path, payload)

    print(f"\n[DONE] ok={ok} fail={fail} error={err} → {out_path}")
    print(f"  tokens: {usage_acc.total_tokens}  calls: {usage_acc.api_calls}")
    run_summary.finish(
        out_path.with_suffix(".run_summary.json"),
        metadata={
            "api_calls": usage_acc.api_calls,
            "tokens": usage_acc.total_tokens,
            "needs_manual_review": sum(
                1 for row in merged.values() if row.get("needs_manual_review")
            ),
        },
    )

    return 1 if err and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
