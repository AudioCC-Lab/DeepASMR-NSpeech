"""Minimal shared runtime for the public annotation and evaluation stages."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_VISION_MODEL = "Qwen/Qwen3.5-397B-A17B"
PROMPT_MARKER_SYSTEM = "###SYSTEM###"
PROMPT_MARKER_USER = "###USER###"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_env() -> Optional[Path]:
    """Load a local .env without overwriting exported environment variables."""
    candidates = (Path.cwd() / ".env", repo_root() / ".env")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(path, override=False)
        except ImportError:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return path
    return None


def env_model_vision() -> str:
    return (os.environ.get("INFO_FULL_VERIFY_VISION_MODEL") or DEFAULT_VISION_MODEL).strip()


def load_info_full_payload(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {}, [dict(row) for row in raw if isinstance(row, dict)]
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object or list: {path}")
    rows = raw.get("data")
    if rows is None:
        rows = raw.get("events")
    if not isinstance(rows, list):
        raise ValueError(f"Expected data/events list: {path}")
    return dict(raw), [dict(row) for row in rows if isinstance(row, dict)]


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def fill_template(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("{" + key + "}", value)
    return out


def clamp_confidence(value: Any) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(10, number))


def norm_token(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "_", str(value).strip().lower())


def norm_noun(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))
    body_parts = {
        "hand": "hands",
        "finger": "fingers",
        "nail": "nails",
        "fingernail": "fingernails",
        "left hand": "hands",
        "right hand": "hands",
        "human hand": "hands",
    }
    text = body_parts.get(text, text)
    if "3dio" in text:
        return "3dio mic"
    if text in {"mic", "mics", "microphone", "microphones"}:
        return "mic"
    return text or None


def parse_json_response(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _usage_from_response(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0) or prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def call_llm_raw(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    image_path: Optional[Path] = None,
    audio_path: Optional[Path] = None,
    timeout: float = 180.0,
    base_url: str = DEFAULT_SILICONFLOW_BASE_URL,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Tuple[str, Dict[str, int]]:
    parts: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    if audio_path is not None and audio_path.is_file():
        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        parts.insert(0, {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_path.suffix.lstrip(".") or "wav"}})
    if image_path is not None and image_path.is_file():
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    request: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": parts},
        ],
        "timeout": timeout,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "siliconflow.cn" in (base_url or "").lower():
        request["extra_body"] = {"enable_thinking": bool(enable_thinking)}
    response = client.chat.completions.create(**request)
    message = response.choices[0].message
    text = (message.content or getattr(message, "reasoning_content", None) or "").strip()
    if not text:
        raise ValueError("LLM returned empty content")
    return text, _usage_from_response(response)


@dataclass
class TokenUsageAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    by_stage: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add(self, usage: Dict[str, int], *, stage: str = "") -> None:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or 0) or prompt + completion
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.api_calls += 1
        if stage:
            bucket = self.by_stage.setdefault(stage, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0})
            bucket["prompt_tokens"] += prompt
            bucket["completion_tokens"] += completion
            bucket["total_tokens"] += total
            bucket["api_calls"] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "by_stage": dict(self.by_stage),
        }


def estimate_cost_cny(usage: TokenUsageAccumulator, *, input_per_m: float, output_per_m: float) -> Dict[str, float]:
    input_cost = usage.prompt_tokens / 1_000_000 * input_per_m
    output_cost = usage.completion_tokens / 1_000_000 * output_per_m
    return {"input_cny": round(input_cost, 4), "output_cny": round(output_cost, 4), "total_cny": round(input_cost + output_cost, 4)}


def call_llm_json_with_usage(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    usage_acc: TokenUsageAccumulator,
    stage: str = "",
    image_path: Optional[Path] = None,
    timeout: float = 180.0,
    max_retries: int = 3,
    sleep_base: float = 1.0,
    base_url: str = DEFAULT_SILICONFLOW_BASE_URL,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    from openai import APIConnectionError, APITimeoutError, RateLimitError  # type: ignore

    retryable = (APITimeoutError, APIConnectionError, RateLimitError)
    for attempt in range(max_retries + 1):
        try:
            text, usage = call_llm_raw(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                image_path=image_path,
                timeout=timeout,
                base_url=base_url,
                enable_thinking=enable_thinking,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage_acc.add(usage, stage=stage)
            return parse_json_response(text)
        except retryable:
            if attempt >= max_retries:
                raise
            time.sleep(sleep_base * (2**attempt))
    raise RuntimeError("unreachable")


_DISAMBIG_HEADER = re.compile(r"^([a-z][a-z0-9_]*)\s+—")


def load_flat_stage2_class_disambiguation(path: Optional[Path] = None) -> Dict[str, str]:
    source = path or (repo_root() / "vocab" / "stage2_class_disambiguation.txt")
    blocks: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _DISAMBIG_HEADER.match(stripped)
        if match:
            current = match.group(1)
            blocks[current] = [line.rstrip()]
        elif current:
            blocks[current].append(line.rstrip())
    if not blocks:
        raise ValueError(f"No disambiguation blocks found: {source}")
    return {key: "\n".join(value).strip() for key, value in blocks.items()}


def format_disambiguation_for_classes(disambiguation: Dict[str, str], classes: List[str]) -> str:
    return "\n\n".join(disambiguation[c] for c in dict.fromkeys(classes) if c in disambiguation)


@dataclass
class VerbClassLexicon:
    classes: List[str] = field(default_factory=list)
    class_hints: Dict[str, str] = field(default_factory=dict)
    noun_cues: Dict[str, str] = field(default_factory=dict)
    verbs_by_class: Dict[str, List[str]] = field(default_factory=dict)
    verb_to_class: Dict[str, str] = field(default_factory=dict)

    @property
    def all_verbs(self) -> Set[str]:
        return set(self.verb_to_class)

    def resolve_class(self, raw: str) -> Optional[str]:
        token = norm_token(raw)
        return next((value for value in self.classes if norm_token(value) == token), None)

    def resolve_verb(self, raw: str) -> Optional[str]:
        token = norm_token(raw)
        return token if token in self.verb_to_class else None

    def class_of(self, verb: str) -> Optional[str]:
        return self.verb_to_class.get(norm_token(verb))

    def format_class_catalog(self) -> str:
        lines: List[str] = []
        for cls in self.classes:
            lines.append(f"- {cls}")
            if self.class_hints.get(cls):
                lines.append(f"  when_to_use: {self.class_hints[cls]}")
            if self.noun_cues.get(cls):
                lines.append(f"  noun_cues: {self.noun_cues[cls]}")
        return "\n".join(lines)

    def format_merged_verbs_for_classes(self, classes: List[str]) -> Tuple[str, List[str]]:
        merged: List[str] = []
        lines: List[str] = []
        for cls in classes:
            verbs = self.verbs_by_class.get(cls, [])
            lines.append(f"- {cls}: {', '.join(verbs)}")
            for verb in verbs:
                if verb not in merged:
                    merged.append(verb)
        block = "\n".join(lines)
        if merged:
            block += f"\n\nAllowed verbs (pick exactly one): {', '.join(merged)}"
        return block, merged

    def format_step2_verb_block(self, classes: List[str], disambiguation: Dict[str, str]) -> Tuple[str, str, List[str]]:
        candidates, allowed = self.format_merged_verbs_for_classes(classes)
        return candidates, format_disambiguation_for_classes(disambiguation, classes), allowed


def load_verb_class_lexicon(path: Optional[Path] = None) -> VerbClassLexicon:
    source = path or (repo_root() / "vocab" / "asmr_verb_classes_en.csv")
    lexicon = VerbClassLexicon()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cls = (row.get("class") or "").strip()
            verbs = [norm_token(part) for part in re.split(r"[;,]", row.get("verbs") or "") if norm_token(part)]
            if not cls or not verbs:
                continue
            lexicon.classes.append(cls)
            lexicon.class_hints[cls] = (row.get("class_hint") or row.get("when_to_use") or "").strip()
            lexicon.noun_cues[cls] = (row.get("noun_cues") or row.get("noun_hint") or "").strip()
            lexicon.verbs_by_class[cls] = verbs
            for verb in verbs:
                if verb in lexicon.verb_to_class:
                    raise ValueError(f"Verb occurs in multiple classes: {verb}")
                lexicon.verb_to_class[verb] = cls
    if not lexicon.all_verbs:
        raise ValueError(f"Empty verb class table: {source}")
    return lexicon


def load_verb_alias_map(path: Optional[Path] = None, *, approved_only: bool = True) -> Dict[str, str]:
    source = path or (repo_root() / "vocab" / "verb_alias.csv")
    if not source.is_file():
        return {}
    aliases: Dict[str, str] = {}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {str(name or "").strip().lower() for name in reader.fieldnames or []}
        grouped = "group_id" in fields
        for row in reader:
            alias = norm_token(row.get("alias"))
            canonical = norm_token(row.get("group_id") if grouped else row.get("canonical"))
            if not grouped and approved_only and (row.get("status") or "approved").strip().lower() != "approved":
                continue
            if alias and canonical:
                aliases[alias] = canonical
    return aliases


def resolve_verb_lexicon(raw: str, lexicon: VerbClassLexicon) -> Optional[str]:
    return lexicon.resolve_verb(raw)


def resolve_verb_canonical(raw: str, lexicon: VerbClassLexicon, aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    token = norm_token(raw)
    direct = lexicon.resolve_verb(token)
    if direct:
        return direct
    mapped = (aliases or {}).get(token)
    return mapped if mapped and lexicon.resolve_verb(mapped) else None


def add_common_cli(parser: argparse.ArgumentParser) -> None:
    load_project_env()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-json", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sleep-segment", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stems", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)


def stems_filter_from_args(stems: Optional[str]) -> Optional[frozenset[str]]:
    return frozenset(part.strip() for part in stems.split(",") if part.strip()) if stems else None


def make_openai_client(api_key: str, base_url: str) -> Any:
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key.strip(), base_url=(base_url or DEFAULT_SILICONFLOW_BASE_URL).strip())
