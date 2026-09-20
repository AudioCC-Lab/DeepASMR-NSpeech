#!/usr/bin/env python3
"""Run an OpenAI-compatible audio judge on SVO-AQA generated model audio."""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from .loudness import cached_rms_match
except ImportError:  # direct script execution
    from loudness import cached_rms_match

MISSING_LOGPROB = -1e9
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-omni-flash"
SYSTEM_PROMPT = (
    "You are an audio understanding assistant. Answer multiple-choice "
    "questions using only the requested letter."
)


def load_questions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a list or contain questions")
    return rows


def load_subset_ids(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows: Iterable[Any] = payload
    elif isinstance(payload, dict):
        rows = payload.get("questions") or payload.get("question_ids") or []
    else:
        raise ValueError(f"invalid subset JSON: {path}")
    ids = {
        str(row.get("question_id") if isinstance(row, dict) else row)
        for row in rows
    }
    ids.discard("")
    return ids


def build_prompt(question: dict) -> str:
    if str(question.get("prompt") or "").strip():
        return str(question["prompt"]).strip()
    context = question.get("prompt_context") or {}
    options = question.get("options") or []
    option_text = "\n".join(
        f"({row['label']}) {str(row['value']).replace('_', ' ')}" for row in options
    )
    letters = "/".join(str(row["label"]) for row in options)
    subject = context.get("subject_core") or "item"
    obj = context.get("object_core") or "surface"
    task = question.get("task")
    if task == "verb_mc":
        query = f"Subject: {subject}\nObject: {obj}\nWhich action best matches the sound?"
    elif task == "object_material_mc":
        query = (
            f"Subject: {subject}\nAction: {context.get('verb', '')}\nObject: {obj}\n"
            "What is the object's material?"
        )
    else:
        material = context.get("object_material")
        suffix = f" ({str(material).replace('_', ' ')})" if material else ""
        query = (
            f"Subject: {subject}\nAction: {context.get('verb', '')}\n"
            f"Object: {obj}{suffix}\nWhat is the subject's material?"
        )
    return f"{query}\n{option_text}\nAnswer with a single letter ({letters}) only."


def encode_audio(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in {"wav", "mp3", "flac", "m4a", "ogg"}:
        raise ValueError(f"unsupported audio format: {path.suffix}")
    return base64.b64encode(path.read_bytes()).decode("ascii"), suffix


def _first_nonempty_token(logprobs: Any) -> Any:
    if logprobs is None:
        return None
    for token in getattr(logprobs, "content", None) or []:
        if str(getattr(token, "token", "") or "").strip():
            return token
    return None


def first_token_scores(logprobs: Any, valid: set[str]) -> Dict[str, float]:
    scores = {letter: MISSING_LOGPROB for letter in valid}
    token = _first_nonempty_token(logprobs)
    if token is None:
        return scores
    candidates = list(getattr(token, "top_logprobs", None) or []) + [token]
    for candidate in candidates:
        letter = str(getattr(candidate, "token", "") or "").strip().upper()
        value = getattr(candidate, "logprob", None)
        if letter in valid and value is not None:
            scores[letter] = max(scores[letter], float(value))
    return scores


def best_letter(scores: Dict[str, float]) -> Optional[str]:
    if not scores:
        return None
    letter = max(scores, key=scores.get)
    return letter if scores[letter] > -1e8 else None


def call_judge(client: Any, *, model: str, prompt: str, audio_path: Path, top_logprobs: int) -> dict:
    audio, audio_format = encode_audio(audio_path)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio, "format": audio_format}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        temperature=0,
        max_tokens=8,
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    text_parts: list[str] = []
    content: list[Any] = []
    usage: dict = {}
    for chunk in stream:
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                text_parts.append(choice.delta.content)
            new_content = list(getattr(choice.logprobs, "content", None) or []) if choice.logprobs else []
            if new_content:
                if not content or len(new_content) > len(content):
                    content = new_content
                elif len(new_content) == 1:
                    content += new_content
                else:
                    content = new_content
        if getattr(chunk, "usage", None):
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }
    wrapper = type("Logprobs", (), {"content": content})() if content else None
    return {"text": "".join(text_parts).strip(), "logprobs": wrapper, "usage": usage}


def load_existing(path: Path) -> Dict[str, dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["question_id"]): row
        for row in payload.get("results") or []
        if row.get("question_id")
    }


def write_output(path: Path, *, metadata: dict, rows: Dict[str, dict]) -> None:
    ordered = [rows[key] for key in sorted(rows)]
    payload = {"meta": {**metadata, "n_results": len(ordered)}, "results": ordered}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True, help="Generated WAVs named {audio_id}.wav")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--reference-audio-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--model", default=os.environ.get("DASHSCOPE_OMNI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    from openai import OpenAI

    questions = load_questions(args.bank)
    subset_ids = load_subset_ids(args.subset)
    if subset_ids is not None:
        questions = [row for row in questions if str(row.get("question_id")) in subset_ids]
    if args.limit > 0:
        questions = questions[: args.limit]

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    results = load_existing(args.out)
    started = time.time()
    metadata = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "judge_model": args.model,
        "temperature": 0,
        "max_tokens": 8,
        "top_logprobs": args.top_logprobs,
        "scoring": "highest first-nonempty-token log-probability",
        "loudness_matching": "reference RMS at 48 kHz" if args.reference_audio_dir else "none",
        "n_questions": len(questions),
    }
    cache_dir = args.cache_dir or (args.out.parent / "loudness_cache")

    for index, question in enumerate(questions, start=1):
        question_id = str(question["question_id"])
        if question_id in results and not results[question_id].get("error") and not args.overwrite:
            continue
        audio_id = str(question["audio_id"])
        source = args.audio_dir / f"{audio_id}.wav"
        audio_path = source
        loudness_metadata = None
        error = None
        response: dict = {"text": "", "logprobs": None, "usage": {}}
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            if args.reference_audio_dir:
                reference = args.reference_audio_dir / f"{audio_id}.wav"
                if not reference.is_file():
                    raise FileNotFoundError(reference)
                audio_path, loudness_metadata = cached_rms_match(
                    source,
                    reference,
                    cache_dir / f"{audio_id}.wav",
                )
            for attempt in range(1, args.max_attempts + 1):
                try:
                    response = call_judge(
                        client,
                        model=args.model,
                        prompt=build_prompt(question),
                        audio_path=audio_path,
                        top_logprobs=args.top_logprobs,
                    )
                    break
                except Exception:
                    if attempt >= args.max_attempts:
                        raise
                    time.sleep(2 ** (attempt - 1))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        valid = {str(option["label"]).upper() for option in question.get("options") or []}
        scores = first_token_scores(response.get("logprobs"), valid)
        pred_logprob = best_letter(scores)
        generated = str(response.get("text") or "").strip().upper()
        pred_generated = next((char for char in generated if char in valid), None)
        results[question_id] = {
            "question_id": question_id,
            "audio_id": audio_id,
            "task": question.get("task"),
            "gold_label": question.get("gold_label"),
            "pred_label": pred_logprob,
            "pred_logprob": pred_logprob,
            "pred_generated": pred_generated,
            "generated_text": response.get("text"),
            "logprob_scores": scores,
            "loudness": loudness_metadata,
            "usage": response.get("usage") or {},
            "error": error,
        }
        metadata["elapsed_seconds"] = round(time.time() - started, 1)
        write_output(args.out, metadata=metadata, rows=results)
        if index % 10 == 0 or index == len(questions):
            print(f"[{index}/{len(questions)}] {question_id}", flush=True)

    write_output(args.out, metadata=metadata, rows=results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
