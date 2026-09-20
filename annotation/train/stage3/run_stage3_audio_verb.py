#!/usr/bin/env python3
"""
Stage3：Omni 音频模型仅选动词；名词与 class 来自 stage2。

输入：train_events_with_visual.json（需 stage2 asmr_class + subject/object）
输出：增量写回 verb、caption 等字段

依赖均在 train/stage3/ 下，目录可整体迁移。无 GT，不做评测。

用法:
  cd /path/to/train/stage3
  export PYTHONPATH=lib:.

  python run_stage3_audio_verb.py --dry-run --limit 3
  python run_stage3_audio_verb.py --continue-on-error
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STAGE3_DIR = Path(__file__).resolve().parent
REPO_ROOT = STAGE3_DIR.parents[2]
LIB_DIR = STAGE3_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_file import load_env_early  # noqa: E402
from annotation.common.info_common import (  # noqa: E402
    TokenUsageAccumulator,
    atomic_write_json,
    clamp_confidence,
    fill_template,
    format_disambiguation_for_classes,
    load_flat_stage2_class_disambiguation,
    load_project_env,
    load_verb_class_lexicon,
    make_openai_client,
    norm_token,
    parse_json_response,
    resolve_verb_lexicon,
)
from omni_client import (  # noqa: E402
    DEFAULT_BASE_URL,
    call_qwen_omni_with_retries,
    estimate_omni_cost_cny,
    pricing_for_model,
)
from annotation.common.prompt_util import load_marked_prompt  # noqa: E402
from train_events_io import event_key, load_train_events, representative_audio_path  # noqa: E402
from deepasmr_nspeech import RunSummary  # noqa: E402

DEFAULT_OUT = STAGE3_DIR.parent / "train_events_with_visual.json"
DEFAULT_PROMPT = STAGE3_DIR / "prompts" / "audio_verb_only_with_text.txt"
DEFAULT_VERB_CSV = REPO_ROOT / "vocab" / "asmr_verb_classes_en.csv"
DEFAULT_DISAMBIG = REPO_ROOT / "vocab" / "stage2_class_disambiguation.txt"
DEFAULT_MODEL = "qwen3.5-omni-plus"


def event_asmr_class(event: dict) -> str:
    cls = (event.get("asmr_class") or "").strip()
    if cls:
        return cls
    short = event.get("asmr_classes_shortlist") or []
    if isinstance(short, list) and short:
        return str(short[0]).strip()
    return ""


def build_caption(subject: Any, verb: str, obj: Any) -> str:
    parts = []
    if subject:
        parts.append(str(subject).strip())
    if verb:
        parts.append(verb.strip())
    if obj:
        parts.append(str(obj).strip())
    return " ".join(parts)


def needs_stage3(event: dict, *, overwrite: bool, skip_missing_audio: bool) -> bool:
    if overwrite:
        return bool(event_asmr_class(event))
    if not event_asmr_class(event):
        return False
    if event.get("stage3_error"):
        return True
    if (event.get("verb") or "").strip():
        return False
    if skip_missing_audio:
        return representative_audio_path(event) is not None
    return True


def rebuild_total_usage(events: List[dict]) -> TokenUsageAccumulator:
    acc = TokenUsageAccumulator()
    for event in events:
        usage = event.get("stage3_usage")
        if isinstance(usage, dict):
            acc.add(usage, stage="stage3_verb")
    return acc


def sync_stage3_meta(
    meta: Dict[str, Any],
    *,
    total_acc: TokenUsageAccumulator,
    model: str,
    total_cost_cny: float,
) -> None:
    meta["stage3_verb_model"] = model
    meta["stage3_verb_usage"] = total_acc.to_dict()
    meta["stage3_verb_pricing_cny_per_m_tokens"] = pricing_for_model(model)
    meta["stage3_verb_cost_cny"] = round(total_cost_cny, 4)
    meta["stage3_verb_updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_verb_context(*, verb_class: str, lex: Any, disambiguation: Dict[str, str]) -> Tuple[str, str, List[str], Optional[str]]:
    cls = lex.resolve_class(verb_class)
    if not cls:
        return "(none)", "(unknown class)", [], None
    candidate_block, allowed = lex.format_merged_verbs_for_classes([cls])
    disambig_block = format_disambiguation_for_classes(disambiguation, [cls])
    return candidate_block, disambig_block, allowed, cls


def process_event(
    *,
    event: dict,
    audio_path: Path,
    lex: Any,
    disambiguation: Dict[str, str],
    system_prompt: str,
    user_tpl: str,
    client: Any,
    model: str,
    args: argparse.Namespace,
) -> dict:
    seg = (event.get("label") or "").strip()
    pred_class = event_asmr_class(event)
    pred_sub = event.get("subject")
    pred_obj = event.get("object")

    candidate_verbs, verb_disambiguation, allowed, cls_resolved = build_verb_context(
        verb_class=pred_class,
        lex=lex,
        disambiguation=disambiguation,
    )
    pred_sub_s = (pred_sub or "").strip() if pred_sub else "(none)"
    pred_obj_s = (pred_obj or "").strip() if pred_obj else "null"

    user_text = fill_template(
        user_tpl,
        {
            "SEG_LABEL": seg or "(none)",
            "VERB_CLASS": cls_resolved or pred_class or "(unknown)",
            "PRED_SUBJECT": pred_sub_s,
            "PRED_OBJECT": pred_obj_s,
            "CANDIDATE_VERBS": candidate_verbs or "(none)",
            "VERB_DISAMBIGUATION": verb_disambiguation or "(none)",
        },
    )

    if args.dry_run:
        out = dict(event)
        out["stage3_dry_run"] = True
        out["stage3_audio_path"] = str(audio_path)
        return out

    raw_text, usage = call_qwen_omni_with_retries(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        audio_path=audio_path,
        enable_thinking=bool(args.enable_thinking),
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        max_retries=int(args.max_retries),
        sleep_base=float(args.sleep_base),
    )
    parsed = parse_json_response(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")

    verb_raw = (parsed.get("verb") or "").strip()
    verb = resolve_verb_lexicon(verb_raw, lex) or norm_token(verb_raw)
    reason = (parsed.get("reason") or parsed.get("llm_verify_reason") or "").strip()
    conf = clamp_confidence(parsed.get("llm_verify_confidence", parsed.get("confidence", 5)))
    audio_class_ok = parsed.get("audio_class_ok")
    verb_invalid = verb not in set(allowed)

    out = dict(event)
    out.update(
        {
            "verb": verb,
            "caption": build_caption(pred_sub, verb, pred_obj),
            "stage3_verb_raw": verb_raw,
            "stage3_reason": reason[:500],
            "stage3_confidence": conf,
            "stage3_audio_class_ok": audio_class_ok,
            "stage3_verb_invalid": verb_invalid,
            "stage3_allowed_verbs": allowed,
            "stage3_audio_path": str(audio_path),
            "stage3_usage": usage,
            "stage3_cost_cny": estimate_omni_cost_cny(usage, model=model),
            "stage3_error": None,
        }
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage3: Omni 仅选动词（无 GT 评测）")
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--verb-classes-csv", type=Path, default=DEFAULT_VERB_CSV)
    p.add_argument("--disambiguation-txt", type=Path, default=DEFAULT_DISAMBIG)
    p.add_argument("--model", default=os.environ.get("DASHSCOPE_OMNI_MODEL") or DEFAULT_MODEL)
    p.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--stems", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--skip-missing-audio", action="store_true", default=True)
    p.add_argument("--require-audio", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep-base", type=float, default=1.5)
    p.add_argument("--sleep-after", type=float, default=0.0)
    p.add_argument("--env-file", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env_paths = [STAGE3_DIR / ".env", STAGE3_DIR.parent.parent / ".env"]
    if args.env_file:
        load_env_early(["--env-file", str(args.env_file)], env_paths)
    else:
        load_env_early(sys.argv, env_paths)
    load_project_env()

    api_key = (args.api_key or os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not args.dry_run and not api_key:
        print("[ERROR] 请设置 DASHSCOPE_API_KEY 或 --dry-run", file=sys.stderr)
        return 1

    if not args.out_json.is_file():
        print(f"[ERROR] 输入不存在: {args.out_json}", file=sys.stderr)
        return 1

    wrapper, events = load_train_events(args.out_json)
    meta = wrapper.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        wrapper["meta"] = meta

    lex = load_verb_class_lexicon(args.verb_classes_csv)
    disambiguation = load_flat_stage2_class_disambiguation(args.disambiguation_txt)
    system_prompt, user_tpl = load_marked_prompt(args.prompt)
    stems = set(args.stems) if args.stems else None
    run_summary = RunSummary("annotation.train.stage3_audio_verb")

    pending: List[int] = []
    for idx, event in enumerate(events):
        key = event_key(event)
        if stems is not None and key not in stems:
            run_summary.add("skipped", item_id=key, reason="not_selected")
            continue
        if idx < args.offset:
            run_summary.add("skipped", item_id=key, reason="before_offset")
            continue
        if needs_stage3(
            event,
            overwrite=args.overwrite,
            skip_missing_audio=args.skip_missing_audio,
        ):
            pending.append(idx)
        elif not event_asmr_class(event):
            run_summary.add("skipped", item_id=key, reason="missing_stage2_class")
        elif representative_audio_path(event) is None:
            run_summary.add("skipped", item_id=key, reason="missing_audio")
        else:
            run_summary.add("skipped", item_id=key, reason="already_completed")

    print(
        f"[INFO] events={len(events)} pending={len(pending)} model={args.model} "
        f"out={args.out_json}"
    )

    if args.dry_run:
        for idx in pending[: min(3, len(pending))]:
            e = events[idx]
            ap = representative_audio_path(e)
            print(f"  sample {event_key(e)} audio={ap}")
        print(f"[DRY-RUN] would call API {len(pending)} times")
        return 0

    if not pending:
        print("[INFO] 无需调用 API")
        run_summary.finish(args.out_json.with_suffix(".stage3.run_summary.json"))
        return 0

    client = make_openai_client(api_key, args.base_url)
    total_acc = rebuild_total_usage(events)
    session_acc = TokenUsageAccumulator()
    total_cost = float(meta.get("stage3_verb_cost_cny") or 0.0)
    n_ok = n_err = n_skip = 0
    limit = args.limit if args.limit > 0 else len(pending)
    called = 0

    for idx, event in enumerate(events):
        key = event_key(event)
        if stems is not None and key not in stems:
            continue
        if idx < args.offset:
            continue
        if not needs_stage3(
            event,
            overwrite=args.overwrite,
            skip_missing_audio=args.skip_missing_audio,
        ):
            n_skip += 1
            continue
        if called >= limit:
            break

        audio_path = representative_audio_path(event)
        if audio_path is None or not audio_path.is_file():
            if args.require_audio:
                n_err += 1
                failed = dict(event)
                failed["stage3_error"] = "missing audio"
                events[idx] = failed
                run_summary.add("failed", item_id=key, reason="missing_audio", detail="missing audio")
                continue
            n_skip += 1
            print(f"[SKIP-NO-AUDIO] {key}")
            run_summary.add("skipped", item_id=key, reason="missing_audio")
            continue

        called += 1
        print(f"[{called}/{limit}] {key} audio={audio_path.name}")
        try:
            updated = process_event(
                event=event,
                audio_path=audio_path,
                lex=lex,
                disambiguation=disambiguation,
                system_prompt=system_prompt,
                user_tpl=user_tpl,
                client=client,
                model=args.model,
                args=args,
            )
            usage = updated.get("stage3_usage")
            if isinstance(usage, dict):
                session_acc.add(usage, stage="stage3_verb")
                total_acc.add(usage, stage="stage3_verb")
            cost = updated.get("stage3_cost_cny") or {}
            if isinstance(cost, dict):
                total_cost += float(cost.get("total_cny") or 0)
            events[idx] = updated
            n_ok += 1
            run_summary.add("success", item_id=key)
            print(f"  verb={updated.get('verb')} caption={updated.get('caption')!r}")
        except Exception as e:
            n_err += 1
            failed = dict(event)
            failed["stage3_error"] = str(e)
            events[idx] = failed
            run_summary.add("failed", item_id=key, detail=e)
            print(f"  [ERROR] {e}", file=sys.stderr)
            if not args.continue_on_error:
                break

        wrapper["events"] = events
        sync_stage3_meta(meta, total_acc=total_acc, model=args.model, total_cost_cny=total_cost)
        if args.checkpoint_every > 0 and called % args.checkpoint_every == 0:
            atomic_write_json(args.out_json, wrapper)

        if args.sleep_after > 0:
            time.sleep(args.sleep_after)

    wrapper["events"] = events
    sync_stage3_meta(meta, total_acc=total_acc, model=args.model, total_cost_cny=total_cost)
    meta["stage3_verb_session"] = {
        "n_ok": n_ok,
        "n_error": n_err,
        "n_skip": n_skip,
        "session_usage": session_acc.to_dict(),
    }
    atomic_write_json(args.out_json, wrapper)

    print(
        f"[DONE] ok={n_ok} err={n_err} skip={n_skip} -> {args.out_json}\n"
        f"  session tokens={session_acc.total_tokens}"
    )
    run_summary.finish(
        args.out_json.with_suffix(".stage3.run_summary.json"),
        metadata={"api_calls": session_acc.api_calls, "tokens": session_acc.total_tokens},
    )
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
