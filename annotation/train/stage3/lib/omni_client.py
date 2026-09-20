"""百炼 Qwen-Omni 系列 OpenAI 兼容调用（音频 + 文本 → 纯文本输出）。"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from annotation.common.info_common import PROMPT_MARKER_SYSTEM, PROMPT_MARKER_USER

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 百炼中国内地 ¥/M tokens（2026-06 官网，多模态输入场景文本输出）
PRICING_CNY_PER_M: Dict[str, Dict[str, float]] = {
    "qwen3-omni-flash": {
        "text_input": 1.8,
        "audio_input": 15.8,
        "text_output_multimodal": 12.7,
    },
    "qwen3.5-omni-plus": {
        "text_input": 1.8,
        "audio_input": 15.8,
        "text_output_multimodal": 12.7,
    },
    "qwen3.5-omni-flash": {
        "text_input": 1.8,
        "audio_input": 15.8,
        "text_output_multimodal": 12.7,
    },
}


def _is_prompt_comment_line(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("#") and not s.startswith("###")


def load_marked_prompt(path: Path) -> Tuple[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if not _is_prompt_comment_line(ln)]
    body = "\n".join(lines).strip()
    if PROMPT_MARKER_USER not in body:
        raise ValueError(f"prompt 缺少 {PROMPT_MARKER_USER}: {path}")
    system_part, user_part = body.split(PROMPT_MARKER_USER, 1)
    system_part = system_part.strip()
    if system_part.startswith(PROMPT_MARKER_SYSTEM):
        system_part = system_part[len(PROMPT_MARKER_SYSTEM) :].lstrip()
    return system_part.strip(), user_part.strip()


def encode_audio_data_url(audio_path: Path) -> Tuple[str, str]:
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    fmt = audio_path.suffix.lstrip(".").lower() or "wav"
    return f"data:;base64,{b64}", fmt


def _usage_dict_from_obj(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    out: Dict[str, Any] = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    ptd = getattr(usage, "prompt_tokens_details", None)
    ctd = getattr(usage, "completion_tokens_details", None)
    if ptd is not None:
        out["prompt_tokens_details"] = (
            dict(ptd) if isinstance(ptd, dict) else _obj_to_dict(ptd)
        )
    if ctd is not None:
        out["completion_tokens_details"] = (
            dict(ctd) if isinstance(ctd, dict) else _obj_to_dict(ctd)
        )
    return out


def _obj_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    d: Dict[str, Any] = {}
    for key in ("text_tokens", "audio_tokens", "video_tokens", "image_tokens", "reasoning_tokens"):
        val = getattr(obj, key, None)
        if val is not None:
            d[key] = val
    return d


def pricing_for_model(model: str) -> Dict[str, float]:
    m = (model or "").lower()
    for key, rates in PRICING_CNY_PER_M.items():
        if key in m:
            return dict(rates)
    return dict(PRICING_CNY_PER_M["qwen3-omni-flash"])


def estimate_omni_cost_cny(usage: Dict[str, Any], *, model: str) -> Dict[str, float]:
    rates = pricing_for_model(model)
    ptd = usage.get("prompt_tokens_details") or {}
    if isinstance(ptd, dict):
        text_in = int(ptd.get("text_tokens") or 0)
        audio_in = int(ptd.get("audio_tokens") or 0)
    else:
        text_in = int(getattr(ptd, "text_tokens", 0) or 0)
        audio_in = int(getattr(ptd, "audio_tokens", 0) or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    if text_in == 0 and audio_in == 0:
        text_in = prompt
    elif text_in + audio_in < prompt:
        text_in += prompt - text_in - audio_in

    ctd = usage.get("completion_tokens_details") or {}
    text_out = int((ctd.get("text_tokens") if isinstance(ctd, dict) else getattr(ctd, "text_tokens", 0)) or 0)
    if text_out == 0:
        text_out = completion

    inp_cny = (
        text_in / 1_000_000 * rates["text_input"]
        + audio_in / 1_000_000 * rates["audio_input"]
    )
    out_cny = text_out / 1_000_000 * rates["text_output_multimodal"]
    return {
        "input_cny": round(inp_cny, 4),
        "output_cny": round(out_cny, 4),
        "total_cny": round(inp_cny + out_cny, 4),
        "text_input_tokens": text_in,
        "audio_input_tokens": audio_in,
        "text_output_tokens": text_out,
    }


def call_qwen_omni_text(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    audio_path: Path,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    """百炼 Qwen-Omni：stream=True，modalities=[text]，返回纯文本。"""
    audio_data, fmt = encode_audio_data_url(audio_path)
    parts: List[Dict[str, Any]] = [
        {"type": "input_audio", "input_audio": {"data": audio_data, "format": fmt}},
        {"type": "text", "text": user_text},
    ]
    req: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": parts},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if enable_thinking:
        req["extra_body"] = {"enable_thinking": True}

    content_parts: List[str] = []
    usage: Dict[str, Any] = {}
    stream = client.chat.completions.create(**req)
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None) or ""
            if piece:
                content_parts.append(piece)
        if getattr(chunk, "usage", None) is not None:
            usage = _usage_dict_from_obj(chunk.usage)

    text = "".join(content_parts).strip()
    if not text:
        raise ValueError("模型返回空 content")
    return text, usage


_OMNI_QUOTA_ERROR_MARKERS = (
    "arrearage",
    "insufficient",
    "quota",
    "额度",
    "allocation quota",
    "exceeded",
    "free tier",
    "欠费",
    "余额",
    "billing",
)


def is_omni_quota_error(exc: BaseException) -> bool:
    """百炼模型/账户额度用尽、欠费等不可在同模型上重试的错误。"""
    s = str(exc).lower()
    return any(marker in s for marker in _OMNI_QUOTA_ERROR_MARKERS)


def build_omni_model_chain(primary: str, backup_models: Optional[List[str]] = None) -> List[str]:
    chain: List[str] = []
    for raw in [primary, *(backup_models or [])]:
        m = (raw or "").strip()
        if m and m not in chain:
            chain.append(m)
    return chain


def _call_qwen_omni_single_model_with_retries(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    audio_path: Path,
    enable_thinking: bool,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    sleep_base: float,
) -> Tuple[str, Dict[str, Any]]:
    from openai import APIConnectionError, APITimeoutError, RateLimitError  # type: ignore

    retryable = (APITimeoutError, APIConnectionError, RateLimitError)
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return call_qwen_omni_text(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                audio_path=audio_path,
                enable_thinking=enable_thinking,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except retryable as e:
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(sleep_base * (2**attempt))
        except Exception as e:
            if is_omni_quota_error(e):
                raise
            raise
    assert last_err is not None
    raise last_err


def call_qwen_omni_with_retries(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    audio_path: Path,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 3,
    sleep_base: float = 1.5,
    backup_models: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """调用 Omni；若主模型额度用尽且提供 backup_models，自动切到下一个模型重试本条。"""
    chain = build_omni_model_chain(model, backup_models)
    last_err: Optional[BaseException] = None
    for idx, active_model in enumerate(chain):
        try:
            text, usage = _call_qwen_omni_single_model_with_retries(
                client=client,
                model=active_model,
                system_prompt=system_prompt,
                user_text=user_text,
                audio_path=audio_path,
                enable_thinking=enable_thinking,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                sleep_base=sleep_base,
            )
            usage = dict(usage)
            usage["omni_model_used"] = active_model
            if idx > 0:
                usage["omni_model_failover_from"] = chain[0]
            return text, usage
        except Exception as e:
            last_err = e
            if is_omni_quota_error(e) and idx < len(chain) - 1:
                nxt = chain[idx + 1]
                print(
                    f"[OMNI-FAILOVER] {active_model} 额度/余额不足，切换至 {nxt}",
                    flush=True,
                )
                continue
            raise
    assert last_err is not None
    raise last_err
