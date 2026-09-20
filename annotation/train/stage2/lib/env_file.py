"""
从 .env 读取 KEY=VALUE 写入 os.environ（不覆盖已在环境中存在的同名变量）。

支持的键（与现有脚本一致，可在 .env 中任选其一）：
  - OPENAI_BASE_URL 或 SILICONFLOW_BASE_URL（后者在 OPENAI_BASE_URL 未设时映射过去）
  - SILICONFLOW_API_KEY 或 OPENAI_API_KEY
  - LABEL_PARSE_MODEL 或 SILICONFLOW_MODEL 或 MODEL（后两者在 LABEL_PARSE_MODEL 未设时映射）

在 argparse 解析前调用 load_env_early，以便 --api-key / --base-url 等默认值能读到 .env。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence


def peek_env_file_arg(argv: Sequence[str]) -> Optional[Path]:
    """从 argv 读取 --env-file PATH（若存在）。"""
    for i, a in enumerate(argv):
        if a == "--env-file" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--env-file="):
            return Path(a.split("=", 1)[1].strip())
    return None


def peek_path_flag(argv: Sequence[str], flag: str) -> Optional[Path]:
    """读取 `--flag PATH` 或 `--flag=PATH`（flag 须为完整长选项名，如 --root）。"""
    eq = f"{flag}="
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith(eq):
            return Path(a[len(eq) :].strip())
    return None


def _parse_env_line(line: str) -> Optional[tuple[str, str]]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.upper().startswith("EXPORT "):
        s = s[7:].lstrip()
    if "=" not in s:
        return None
    k, _, rest = s.partition("=")
    k = k.strip()
    if not k:
        return None
    v = rest.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return k, v


def load_env_file(path: Path) -> int:
    """
    将 path 中定义的变量写入 os.environ（仅当该键尚不存在于 os.environ 时）。
    返回成功解析并尝试设置的键数量（不含已跳过已存在键）。
    """
    if not path.is_file():
        return 0
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    n = 0
    for line in raw.splitlines():
        pair = _parse_env_line(line)
        if not pair:
            continue
        k, v = pair
        if k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def apply_llm_env_aliases() -> None:
    """把硅基流动常用别名映射到 OpenAI SDK / 脚本使用的变量名。"""
    sil_url = (os.environ.get("SILICONFLOW_BASE_URL") or "").strip()
    if sil_url and not (os.environ.get("OPENAI_BASE_URL") or "").strip():
        os.environ["OPENAI_BASE_URL"] = sil_url

    model = (
        (os.environ.get("SILICONFLOW_MODEL") or "").strip()
        or (os.environ.get("MODEL") or "").strip()
    )
    if model and not (os.environ.get("LABEL_PARSE_MODEL") or "").strip():
        os.environ["LABEL_PARSE_MODEL"] = model


def load_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        rp = p.expanduser().resolve()
        if rp.is_file():
            load_env_file(rp)
            return rp
    return None


def load_env_early(argv: Sequence[str], default_paths: Sequence[Path]) -> Optional[Path]:
    """
    优先加载 --env-file 指定路径；否则加载 default_paths 中第一个存在的文件。
    随后 apply_llm_env_aliases。返回实际加载的 .env 路径，未加载则 None。
    """
    peeked = peek_env_file_arg(argv)
    loaded: Optional[Path] = None
    if peeked is not None:
        rp = peeked.expanduser().resolve()
        if rp.is_file():
            load_env_file(rp)
            loaded = rp
        else:
            print(f"[WARN] --env-file 不存在: {rp}，将尝试默认路径", file=sys.stderr)
            loaded = load_first_existing(default_paths)
    else:
        loaded = load_first_existing(default_paths)
    apply_llm_env_aliases()
    if loaded:
        print(f"[INFO] 已加载 .env: {loaded}", file=sys.stderr)
    return loaded
