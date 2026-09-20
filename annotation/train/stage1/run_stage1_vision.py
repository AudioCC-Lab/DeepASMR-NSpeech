#!/usr/bin/env python3
"""
Stage 1 视觉：为 train_events.json 中每个 event 写入 visual_description。

每个 event 对应 fig_train 中一张关键帧（{id}_{start_sec}_{end_sec}.png），
同一 event 下各 clip 共用 event 级 label，因此每 event 只调一次视觉 API。

用法:
  python annotation/train/stage1/run_stage1_vision.py --dry-run --limit 10

  # 试跑 10 条（推荐关 thinking 省 output token）
  python run_stage1_vision.py --limit 10 --no-enable-thinking

  # 全量 + 断点续跑
  python run_stage1_vision.py --no-enable-thinking --continue-on-error
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

STAGE1_DIR = Path(__file__).resolve().parent
REPO_ROOT = STAGE1_DIR.parents[2]
ANNOTATION_ROOT = STAGE1_DIR.parents[1]
LIB_DIR = STAGE1_DIR.parent / "stage2" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from filter_blocked_train_events import filter_events  # noqa: E402
from deepasmr_nspeech import RunSummary  # noqa: E402

from annotation.common.info_common import (  # noqa: E402
    TokenUsageAccumulator,
    add_common_cli,
    atomic_write_json,
    call_llm_raw,
    env_model_vision,
    estimate_cost_cny,
    fill_template,
    load_project_env,
    make_openai_client,
    stems_filter_from_args,
)

# 无历史样本时的 token 估算（MiMo/Qwen 相近图尺寸实测均值）
DEFAULT_EST_PROMPT_TOKENS = 668
DEFAULT_EST_COMPLETION_TOKENS = 501

DEFAULT_INPUT = ANNOTATION_ROOT / "train" / "train_events.json"
DEFAULT_OUTPUT = ANNOTATION_ROOT / "train" / "train_events_with_visual.json"
DEFAULT_FIGS = ANNOTATION_ROOT / "train" / "frames"
DEFAULT_SYSTEM_PROMPT = STAGE1_DIR / "prompts" / "stage1_system.txt"
DEFAULT_USER_PROMPT = STAGE1_DIR / "prompts" / "stage1_user.txt"

# 硅基流动 Qwen3.5-397B-A17B（2026-06 官网）
DEFAULT_PRICING_CNY = {"input": 1.2, "output": 7.2}


def load_prompt_files(system_path: Path, user_path: Path) -> Tuple[str, str]:
    if not system_path.is_file() or not user_path.is_file():
        raise FileNotFoundError(
            f"缺少 prompt: system={system_path} user={user_path}"
        )
    return (
        system_path.read_text(encoding="utf-8").strip(),
        user_path.read_text(encoding="utf-8").strip(),
    )


def event_key(event: dict) -> str:
    return f"{event['id']}_{int(float(event['start_sec']))}_{int(float(event['end_sec']))}"


def event_fig_path(event: dict, figs_dir: Path) -> Path:
    return figs_dir / f"{event_key(event)}.png"


def load_train_events(path: Path) -> Tuple[Dict[str, Any], List[dict]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} 需为 {{events: [...]}} 对象")
    events = obj.get("events")
    if not isinstance(events, list):
        raise ValueError(f"{path} 缺少 events 数组")
    return obj, [dict(e) for e in events if isinstance(e, dict)]


def apply_blocked_event_filter(
    wrapper: Dict[str, Any], events: List[dict]
) -> Tuple[List[dict], List[dict]]:
    kept, removed = filter_events(events)
    if removed:
        wrapper["events"] = kept
        if "event_count" in wrapper:
            wrapper["event_count"] = len(kept)
        meta = wrapper.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["blocked_events_removed"] = len(removed)
            meta["blocked_events_removed_labels"] = sorted(
                {str(e.get("label") or "") for e in removed}
            )
    return kept, removed


def load_working_train_events(*, out_path: Path, source_path: Path) -> Tuple[Dict[str, Any], List[dict]]:
    if out_path.is_file():
        wrapper, events = load_train_events(out_path)
    else:
        if not source_path.is_file():
            raise FileNotFoundError(f"源文件不存在: {source_path}")
        wrapper, events = load_train_events(source_path)
        wrapper = copy.deepcopy(wrapper)
        wrapper.setdefault("meta", {})
        meta = wrapper["meta"]
        if not isinstance(meta, dict):
            meta = {}
            wrapper["meta"] = meta
        meta.setdefault("stage1_visual_source_json", str(source_path))
        meta["stage1_visual_initialized_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        wrapper["events"] = events
        atomic_write_json(out_path, wrapper)
    events, removed = apply_blocked_event_filter(wrapper, events)
    if removed:
        atomic_write_json(out_path, wrapper)
        print(
            f"[INFO] removed {len(removed)} blocked events "
            f"(preview/intro/combination/medley-mix) on load"
        )
    return wrapper, events


def needs_stage1(event: dict, *, overwrite: bool) -> bool:
    if overwrite:
        return True
    return not (event.get("visual_description") or "").strip()


def rebuild_total_usage_from_events(events: List[dict]) -> TokenUsageAccumulator:
    acc = TokenUsageAccumulator()
    for event in events:
        usage = event.get("visual_usage")
        if isinstance(usage, dict):
            acc.add(usage, stage="vision")
    return acc


def load_total_usage(meta: Dict[str, Any], events: List[dict]) -> TokenUsageAccumulator:
    """优先从各 event 的 visual_usage 汇总，避免断点续跑丢失累计统计。"""
    from_events = rebuild_total_usage_from_events(events)
    from_meta = usage_acc_from_dict(meta.get("stage1_visual_usage") or {})
    if from_events.api_calls >= from_meta.api_calls:
        return from_events
    return from_meta


def usage_acc_from_dict(data: Dict[str, Any]) -> TokenUsageAccumulator:
    acc = TokenUsageAccumulator()
    if not isinstance(data, dict):
        return acc
    acc.prompt_tokens = int(data.get("prompt_tokens") or 0)
    acc.completion_tokens = int(data.get("completion_tokens") or 0)
    acc.total_tokens = int(data.get("total_tokens") or 0) or (
        acc.prompt_tokens + acc.completion_tokens
    )
    acc.api_calls = int(data.get("api_calls") or 0)
    by_stage = data.get("by_stage")
    if isinstance(by_stage, dict):
        acc.by_stage = {
            str(k): dict(v) if isinstance(v, dict) else {}
            for k, v in by_stage.items()
        }
    return acc


def avg_usage_from_events(events: List[dict]) -> Tuple[float, float]:
    prompts: List[int] = []
    completions: List[int] = []
    for event in events:
        usage = event.get("visual_usage")
        if not isinstance(usage, dict):
            continue
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        if p > 0 or c > 0:
            prompts.append(p)
            completions.append(c)
    if prompts:
        return sum(prompts) / len(prompts), sum(completions) / len(completions)
    return float(DEFAULT_EST_PROMPT_TOKENS), float(DEFAULT_EST_COMPLETION_TOKENS)


def estimate_pending_cost(
    *,
    n_calls: int,
    avg_prompt: float,
    avg_completion: float,
    pricing: Dict[str, float],
) -> Dict[str, Any]:
    prompt_tokens = int(round(avg_prompt * n_calls))
    completion_tokens = int(round(avg_completion * n_calls))
    acc = TokenUsageAccumulator()
    acc.prompt_tokens = prompt_tokens
    acc.completion_tokens = completion_tokens
    acc.total_tokens = prompt_tokens + completion_tokens
    acc.api_calls = n_calls
    return {
        "n_calls": n_calls,
        "avg_prompt_tokens": round(avg_prompt, 1),
        "avg_completion_tokens": round(avg_completion, 1),
        "estimated_usage": acc.to_dict(),
        "estimated_cost_cny": estimate_cost_cny(
            acc,
            input_per_m=pricing["input"],
            output_per_m=pricing["output"],
        ),
    }


def sync_usage_meta(
    meta: Dict[str, Any],
    *,
    total_acc: TokenUsageAccumulator,
    session_acc: TokenUsageAccumulator,
    pricing: Dict[str, float],
    model: str,
) -> None:
    meta["stage1_visual_model"] = model
    meta["stage1_visual_usage"] = total_acc.to_dict()
    meta["stage1_visual_usage_session"] = session_acc.to_dict()
    meta["stage1_visual_pricing_cny_per_m_tokens"] = pricing
    meta["stage1_visual_cost_cny"] = estimate_cost_cny(
        total_acc,
        input_per_m=pricing["input"],
        output_per_m=pricing["output"],
    )
    meta["stage1_visual_cost_session_cny"] = estimate_cost_cny(
        session_acc,
        input_per_m=pricing["input"],
        output_per_m=pricing["output"],
    )


def format_usage_cost(
    acc: TokenUsageAccumulator,
    pricing: Dict[str, float],
    *,
    prefix: str = "",
) -> str:
    cost = estimate_cost_cny(
        acc,
        input_per_m=pricing["input"],
        output_per_m=pricing["output"],
    )
    return (
        f"{prefix}tokens in={acc.prompt_tokens} out={acc.completion_tokens} "
        f"total={acc.total_tokens} calls={acc.api_calls} "
        f"cost=¥{cost['input_cny']:.4f}+¥{cost['output_cny']:.4f}=¥{cost['total_cny']:.4f}"
    )


def print_usage_summary(
    *,
    total_acc: TokenUsageAccumulator,
    session_acc: TokenUsageAccumulator,
    pricing: Dict[str, float],
) -> None:
    print("\n=== Token / 费用统计 ===")
    print(f"  累计  {format_usage_cost(total_acc, pricing)}")
    print(f"  本次  {format_usage_cost(session_acc, pricing)}")


def checkpoint_maybe(
    *,
    wrapper: dict,
    out_path: Path,
    called: int,
    every: int,
    meta: Dict[str, Any],
    total_acc: TokenUsageAccumulator,
    session_acc: TokenUsageAccumulator,
    pricing: Dict[str, float],
    model: str,
) -> None:
    sync_usage_meta(
        meta,
        total_acc=total_acc,
        session_acc=session_acc,
        pricing=pricing,
        model=model,
    )
    if every > 0 and called > 0 and called % every == 0:
        atomic_write_json(out_path, wrapper)


def main() -> int:
    load_project_env()
    ap = argparse.ArgumentParser(description="Stage1: train_events 视觉描述（每 event 一张关键帧）")
    add_common_cli(ap)
    ap.set_defaults(source_json=DEFAULT_INPUT, out_json=DEFAULT_OUTPUT)
    ap.add_argument("--figs-dir", type=Path, default=DEFAULT_FIGS)
    ap.add_argument("--system-prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--user-prompt-file", type=Path, default=DEFAULT_USER_PROMPT)
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖 INFO_FULL_VERIFY_VISION_MODEL（默认 Qwen/Qwen3.5-397B-A17B）",
    )
    ap.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="默认关闭；Stage1 只需一段英文，关 thinking 可显著降低 output token 费用",
    )
    ap.add_argument("--skip-missing-image", action="store_true", default=True)
    ap.add_argument("--require-image", action="store_true")
    ap.add_argument(
        "--price-input-cny",
        type=float,
        default=DEFAULT_PRICING_CNY["input"],
        help="输入 token 单价（¥/M tokens）",
    )
    ap.add_argument(
        "--price-output-cny",
        type=float,
        default=DEFAULT_PRICING_CNY["output"],
        help="输出 token 单价（¥/M tokens）",
    )
    args = ap.parse_args()

    model = (args.model or "").strip() or env_model_vision()
    pricing = {
        "input": float(args.price_input_cny),
        "output": float(args.price_output_cny),
    }
    input_path = (args.source_json or DEFAULT_INPUT).resolve()
    out_path = (args.out_json or DEFAULT_OUTPUT).resolve()
    figs_dir = args.figs_dir.resolve()
    system_prompt_path = args.system_prompt_file.resolve()
    user_prompt_path = args.user_prompt_file.resolve()
    stems = stems_filter_from_args(args.stems)

    try:
        system_prompt, user_tpl = load_prompt_files(system_prompt_path, user_prompt_path)
    except FileNotFoundError as e:
        print(f"[ERR] {e}", file=sys.stderr)
        return 1

    wrapper, events = load_working_train_events(out_path=out_path, source_path=input_path)
    meta = wrapper.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        wrapper["meta"] = meta

    run_summary = RunSummary("annotation.train.stage1_vision")
    pending: List[int] = []
    for idx, event in enumerate(events):
        key = event_key(event)
        if stems is not None and key not in stems:
            run_summary.add("skipped", item_id=key, reason="not_selected")
            continue
        if not needs_stage1(event, overwrite=args.overwrite):
            run_summary.add("skipped", item_id=key, reason="already_completed")
            continue
        fig = event_fig_path(event, figs_dir)
        if args.require_image and not fig.is_file():
            run_summary.add("skipped", item_id=key, reason="missing_image")
            continue
        if args.skip_missing_image and not fig.is_file():
            run_summary.add("skipped", item_id=key, reason="missing_image")
            continue
        pending.append(idx)

    avg_prompt, avg_completion = avg_usage_from_events(events)
    print(
        f"[INFO] events={len(events)} need={len(pending)} model={model} "
        f"thinking={args.enable_thinking} figs={figs_dir} out={out_path}"
    )
    print(
        f"[INFO] pricing in=¥{pricing['input']}/M out=¥{pricing['output']}/M "
        f"est_avg={avg_prompt:.0f}+{avg_completion:.0f} tok/call"
    )
    if args.dry_run:
        est = estimate_pending_cost(
            n_calls=len(pending),
            avg_prompt=avg_prompt,
            avg_completion=avg_completion,
            pricing=pricing,
        )
        cost = est["estimated_cost_cny"]
        print(f"[DRY-RUN] API calls ≈ {len(pending)}")
        print(
            f"[DRY-RUN] 预估 tokens in={est['estimated_usage']['prompt_tokens']} "
            f"out={est['estimated_usage']['completion_tokens']} "
            f"total={est['estimated_usage']['total_tokens']}"
        )
        print(
            f"[DRY-RUN] 预估费用 ≈ ¥{cost['total_cny']:.2f} "
            f"(in ¥{cost['input_cny']:.2f} + out ¥{cost['output_cny']:.2f})"
        )
        if pending[:3]:
            for idx in pending[:3]:
                e = events[idx]
                print(f"  sample key={event_key(e)} label={e.get('label')!r}")
        return 0
    if not pending:
        total_acc = load_total_usage(meta, events)
        if total_acc.api_calls > 0:
            sync_usage_meta(
                meta,
                total_acc=total_acc,
                session_acc=TokenUsageAccumulator(),
                pricing=pricing,
                model=model,
            )
            atomic_write_json(out_path, wrapper)
            print_usage_summary(
                total_acc=total_acc,
                session_acc=TokenUsageAccumulator(),
                pricing=pricing,
            )
        print("[INFO] 无需调用 API")
        run_summary.finish(out_path.with_suffix(".stage1.run_summary.json"))
        return 0
    if not (args.api_key or "").strip():
        print("[ERR] 需要 SILICONFLOW_API_KEY / --api-key", file=sys.stderr)
        return 1

    client = make_openai_client(args.api_key, args.base_url)
    total_acc = load_total_usage(meta, events)
    session_acc = TokenUsageAccumulator()
    if total_acc.api_calls > 0:
        print(format_usage_cost(total_acc, pricing, prefix="[INFO] 已累计 "))
    called = skipped = errors = 0
    limit = max(0, int(args.limit))

    for idx, event in enumerate(events):
        key = event_key(event)
        if stems is not None and key not in stems:
            continue
        if not needs_stage1(event, overwrite=args.overwrite):
            skipped += 1
            continue
        if limit > 0 and called >= limit:
            break

        fig = event_fig_path(event, figs_dir)
        if args.require_image and not fig.is_file():
            skipped += 1
            print(f"[SKIP-NO-IMG] {key}")
            continue
        if args.skip_missing_image and not fig.is_file():
            skipped += 1
            continue

        user_text = fill_template(
            user_tpl,
            {
                "VIDEO_TITLE": (event.get("title") or "").strip(),
                "EVENT_LABEL": (event.get("label") or "").strip(),
            },
        )
        try:
            desc, usage = call_llm_raw(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                image_path=fig,
                timeout=float(args.timeout),
                base_url=args.base_url,
                enable_thinking=bool(args.enable_thinking),
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
            )
            row_usage = dict(usage)
            total_acc.add(row_usage, stage="vision")
            session_acc.add(row_usage, stage="vision")
        except Exception as e:
            errors += 1
            run_summary.add("failed", item_id=key, detail=e)
            print(f"[ERR] {key}: {e}", file=sys.stderr)
            if not args.continue_on_error:
                run_summary.finish(out_path.with_suffix(".stage1.run_summary.json"))
                return 2
            continue

        event = dict(event)
        event["visual_description"] = str(desc).strip()
        event["visual_description_model"] = model
        event["visual_keyframe"] = str(fig.relative_to(figs_dir))
        event["visual_description_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        event["visual_usage"] = row_usage
        events[idx] = event
        called += 1
        run_summary.add("success", item_id=key)
        row_cost = estimate_cost_cny(
            usage_acc_from_dict(row_usage),
            input_per_m=pricing["input"],
            output_per_m=pricing["output"],
        )
        print(
            f"[{called}/{len(pending) if limit <= 0 else min(limit, len(pending))}] "
            f"OK {key} in={row_usage.get('prompt_tokens', 0)} "
            f"out={row_usage.get('completion_tokens', 0)} "
            f"¥{row_cost['total_cny']:.4f} | "
            f"累计¥{estimate_cost_cny(total_acc, input_per_m=pricing['input'], output_per_m=pricing['output'])['total_cny']:.4f}"
        )
        checkpoint_maybe(
            wrapper=wrapper,
            out_path=out_path,
            called=called,
            every=int(args.checkpoint_every),
            meta=meta,
            total_acc=total_acc,
            session_acc=session_acc,
            pricing=pricing,
            model=model,
        )
        if args.sleep_segment > 0:
            time.sleep(float(args.sleep_segment))

    wrapper["events"] = events
    events, removed = apply_blocked_event_filter(wrapper, events)
    if removed:
        print(
            f"[INFO] removed {len(removed)} blocked events "
            f"(preview/intro/combination/medley-mix) before finalize"
        )
    meta["stage1_visual_enable_thinking"] = bool(args.enable_thinking)
    meta["stage1_visual_figs_dir"] = str(args.figs_dir)
    meta["stage1_visual_system_prompt"] = str(args.system_prompt_file)
    meta["stage1_visual_user_prompt"] = str(args.user_prompt_file)
    meta["stage1_visual_finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sync_usage_meta(
        meta,
        total_acc=total_acc,
        session_acc=session_acc,
        pricing=pricing,
        model=model,
    )
    atomic_write_json(out_path, wrapper)

    cost = meta["stage1_visual_cost_cny"]
    session_cost = meta["stage1_visual_cost_session_cny"]
    print(
        f"[DONE] api={called} skipped={skipped} errors={errors} → {out_path}"
    )
    print_usage_summary(
        total_acc=total_acc,
        session_acc=session_acc,
        pricing=pricing,
    )
    print(
        f"[DONE] 累计 ¥{cost.get('total_cny')} | 本次 ¥{session_cost.get('total_cny')}"
    )
    run_summary.finish(
        out_path.with_suffix(".stage1.run_summary.json"),
        metadata={"api_calls": session_acc.api_calls, "tokens": session_acc.total_tokens},
    )
    return 0 if errors == 0 or args.continue_on_error else 2


if __name__ == "__main__":
    raise SystemExit(main())
