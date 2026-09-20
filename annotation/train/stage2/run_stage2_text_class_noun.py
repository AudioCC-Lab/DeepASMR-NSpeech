#!/usr/bin/env python3
"""
Stage2：文本 LLM 一次调用 → verb 大类 top-2 + subject + object。

输入：train_events_with_visual.json（需 stage1 visual_description）
输出：增量写回同一 JSON 的 event 字段

依赖均在 train/stage2/ 下（lib、data、prompts），目录可整体迁移。

用法:
  cd /path/to/train/stage2
  export PYTHONPATH=lib:.

  python run_stage2_text_class_noun.py --dry-run --limit 3
  python run_stage2_text_class_noun.py --continue-on-error
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STAGE2_DIR = Path(__file__).resolve().parent
REPO_ROOT = STAGE2_DIR.parents[2]
LIB_DIR = STAGE2_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_file import load_env_early  # noqa: E402
from annotation.common.info_common import (  # noqa: E402
    DEFAULT_SILICONFLOW_BASE_URL,
    TokenUsageAccumulator,
    atomic_write_json,
    call_llm_json_with_usage,
    clamp_confidence,
    estimate_cost_cny,
    fill_template,
    load_project_env,
    load_verb_class_lexicon,
    make_openai_client,
    norm_noun,
    norm_token,
)
from annotation.common.prompt_util import load_marked_prompt  # noqa: E402
from train_events_io import event_key, load_train_events  # noqa: E402
from deepasmr_nspeech import RunSummary  # noqa: E402

DEFAULT_OUT = STAGE2_DIR.parent / "train_events_with_visual.json"
DEFAULT_PROMPT = STAGE2_DIR / "prompts" / "class_and_noun_joint.txt"
DEFAULT_VERB_CSV = REPO_ROOT / "vocab" / "asmr_verb_classes_en.csv"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
MODEL_PRICING_CNY = {"Qwen/Qwen3.6-35B-A3B": {"input": 0.40, "output": 3.20}}


def parse_classes(parsed: dict, lex: Any) -> List[str]:
    classes_raw = parsed.get("asmr_classes") or parsed.get("verb_classes") or []
    if isinstance(classes_raw, str):
        classes_raw = [classes_raw]
    classes: List[str] = []
    for c in classes_raw:
        resolved = lex.resolve_class(str(c))
        if resolved and resolved not in classes:
            classes.append(resolved)
    if len(classes) < 2 and classes:
        classes = classes + classes[: 2 - len(classes)]
    return classes[:2]


def needs_stage2(event: dict, *, overwrite: bool) -> bool:
    if overwrite:
        return True
    if not (event.get("visual_description") or "").strip():
        return False
    if event.get("stage2_error"):
        return True
    return not (event.get("asmr_class") or "").strip()


def rebuild_total_usage(events: List[dict]) -> TokenUsageAccumulator:
    acc = TokenUsageAccumulator()
    for event in events:
        usage = event.get("stage2_usage")
        if isinstance(usage, dict):
            acc.add(usage, stage="stage2_text")
    return acc


def sync_stage2_meta(meta: Dict[str, Any], *, total_acc: TokenUsageAccumulator, model: str) -> None:
    meta["stage2_text_model"] = model
    meta["stage2_text_usage"] = total_acc.to_dict()
    rates = MODEL_PRICING_CNY.get(model, {"input": 0.40, "output": 3.20})
    meta["stage2_text_cost_cny"] = estimate_cost_cny(
        total_acc, input_per_m=rates["input"], output_per_m=rates["output"]
    )
    meta["stage2_text_updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def process_event(
    *,
    event: dict,
    lex: Any,
    system_prompt: str,
    user_tpl: str,
    client: Any,
    model: str,
    args: argparse.Namespace,
) -> Tuple[dict, Optional[str], Optional[dict]]:
    seg = (event.get("label") or "").strip()
    visual = (event.get("visual_description") or "").strip()
    user_text = fill_template(
        user_tpl,
        {
            "SEG_LABEL": seg or "(none)",
            "VISUAL_DESCRIPTION": visual or "(none)",
            "CLASS_CATALOG": lex.format_class_catalog() or "(none)",
        },
    )
    if args.dry_run:
        event = dict(event)
        event["stage2_dry_run"] = True
        return event, None, None

    usage_one = TokenUsageAccumulator()
    parsed = call_llm_json_with_usage(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        usage_acc=usage_one,
        stage="stage2_text",
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        max_retries=int(args.max_retries),
        sleep_base=float(args.sleep_base),
    )
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")

    classes = parse_classes(parsed, lex)
    obj_raw = parsed.get("object")
    obj = None
    if obj_raw is not None and str(obj_raw).strip().lower() not in ("null", "none", ""):
        obj = norm_noun(obj_raw)

    out = dict(event)
    out.update(
        {
            "subject": norm_noun(parsed.get("subject")),
            "object": obj,
            "asmr_classes_shortlist": classes,
            "asmr_class": classes[0] if classes else None,
            "stage2_text_reason": (parsed.get("llm_verify_reason") or "").strip()[:500],
            "stage2_confidence": clamp_confidence(parsed.get("llm_verify_confidence", 5)),
            "stage2_error": None,
        }
    )
    return out, None, usage_one.to_dict()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2: 文本 LLM → class + subject + object")
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--verb-classes-csv", type=Path, default=DEFAULT_VERB_CSV)
    p.add_argument("--model", default=os.environ.get("MODEL_TEXT") or DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--stems", nargs="*", default=None, help="仅处理指定 event_key")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep-base", type=float, default=1.5)
    p.add_argument("--sleep-after", type=float, default=0.0)
    p.add_argument("--env-file", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env_paths = [STAGE2_DIR / ".env", STAGE2_DIR.parent.parent / ".env"]
    if args.env_file:
        load_env_early(["--env-file", str(args.env_file)], env_paths)
    else:
        load_env_early(sys.argv, env_paths)
    load_project_env()

    api_key = (os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not args.dry_run and not api_key:
        print("[ERROR] 请设置 SILICONFLOW_API_KEY 或 --dry-run", file=sys.stderr)
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
    system_prompt, user_tpl = load_marked_prompt(args.prompt)
    stems = set(args.stems) if args.stems else None
    run_summary = RunSummary("annotation.train.stage2_text_svo")

    pending: List[int] = []
    for idx, event in enumerate(events):
        key = event_key(event)
        if stems is not None and key not in stems:
            run_summary.add("skipped", item_id=key, reason="not_selected")
            continue
        if idx < args.offset:
            run_summary.add("skipped", item_id=key, reason="before_offset")
            continue
        if needs_stage2(event, overwrite=args.overwrite):
            pending.append(idx)
        elif not (event.get("visual_description") or "").strip():
            run_summary.add("skipped", item_id=key, reason="missing_visual_description")
        else:
            run_summary.add("skipped", item_id=key, reason="already_completed")

    print(
        f"[INFO] events={len(events)} pending={len(pending)} model={args.model} "
        f"out={args.out_json}"
    )
    n_vis = sum(1 for e in events if (e.get("visual_description") or "").strip())
    if not pending and n_vis == 0:
        print(
            "[WARN] 0 条待处理：所有 event 的 visual_description 均为空。\n"
            "  Stage2 需要 Stage1 视觉描述。请先跑 Stage1，且 OUT_JSON 指向同一文件，例如：\n"
            f"    {args.out_json}\n"
            "  （Stage1 曾默认写到 train/stage1/train_events_with_visual.json，与 Stage2 路径不一致会导致此情况）",
            file=sys.stderr,
        )

    if args.dry_run:
        for idx in pending[: min(3, len(pending))]:
            e = events[idx]
            print(f"  sample {event_key(e)} label={e.get('label')!r}")
        print(f"[DRY-RUN] would call API {len(pending)} times")
        return 0

    if not pending:
        print("[INFO] 无需调用 API")
        run_summary.finish(args.out_json.with_suffix(".stage2.run_summary.json"))
        return 0

    client = make_openai_client(api_key, DEFAULT_SILICONFLOW_BASE_URL)
    total_acc = rebuild_total_usage(events)
    session_acc = TokenUsageAccumulator()
    n_ok = n_err = n_skip = 0
    limit = args.limit if args.limit > 0 else len(pending)
    called = 0

    for idx, event in enumerate(events):
        key = event_key(event)
        if stems is not None and key not in stems:
            continue
        if idx < args.offset:
            continue
        if not needs_stage2(event, overwrite=args.overwrite):
            n_skip += 1
            continue
        if called >= limit:
            break

        called += 1
        print(f"[{called}/{limit}] {key} label={event.get('label')!r}")
        try:
            updated, err, usage = process_event(
                event=event,
                lex=lex,
                system_prompt=system_prompt,
                user_tpl=user_tpl,
                client=client,
                model=args.model,
                args=args,
            )
            if usage:
                session_acc.add(usage, stage="stage2_text")
                total_acc.add(usage, stage="stage2_text")
                updated["stage2_usage"] = usage
            events[idx] = updated
            n_ok += 1
            run_summary.add("success", item_id=key)
            print(
                f"  class={updated.get('asmr_class')} "
                f"sub={updated.get('subject')} obj={updated.get('object')}"
            )
        except Exception as e:
            n_err += 1
            failed = dict(event)
            failed["stage2_error"] = str(e)
            events[idx] = failed
            run_summary.add("failed", item_id=key, detail=e)
            print(f"  [ERROR] {e}", file=sys.stderr)
            if not args.continue_on_error:
                break

        wrapper["events"] = events
        sync_stage2_meta(meta, total_acc=total_acc, model=args.model)
        if args.checkpoint_every > 0 and called % args.checkpoint_every == 0:
            atomic_write_json(args.out_json, wrapper)

        if args.sleep_after > 0:
            time.sleep(args.sleep_after)

    wrapper["events"] = events
    sync_stage2_meta(meta, total_acc=total_acc, model=args.model)
    meta["stage2_text_session"] = {
        "n_ok": n_ok,
        "n_error": n_err,
        "n_skip": n_skip,
        "session_usage": session_acc.to_dict(),
    }
    atomic_write_json(args.out_json, wrapper)

    rates = MODEL_PRICING_CNY.get(args.model, {"input": 0.40, "output": 3.20})
    cost = estimate_cost_cny(
        session_acc, input_per_m=rates["input"], output_per_m=rates["output"]
    )
    print(
        f"[DONE] ok={n_ok} err={n_err} skip={n_skip} -> {args.out_json}\n"
        f"  session tokens={session_acc.total_tokens} est_cost ¥{cost['total_cny']:.4f}"
    )
    run_summary.finish(
        args.out_json.with_suffix(".stage2.run_summary.json"),
        metadata={"api_calls": session_acc.api_calls, "tokens": session_acc.total_tokens},
    )
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
