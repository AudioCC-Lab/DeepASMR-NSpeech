"""Load prompts split by ``###SYSTEM###`` and ``###USER###`` markers."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

PROMPT_MARKER_SYSTEM = "###SYSTEM###"
PROMPT_MARKER_USER = "###USER###"


def load_marked_prompt(path: Path) -> Tuple[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not (line.lstrip().startswith("#") and not line.lstrip().startswith("###"))
    ]
    body = "\n".join(lines).strip()
    if PROMPT_MARKER_USER not in body:
        raise ValueError(f"prompt is missing {PROMPT_MARKER_USER}: {path}")
    system, user = body.split(PROMPT_MARKER_USER, 1)
    system = system.strip()
    if system.startswith(PROMPT_MARKER_SYSTEM):
        system = system[len(PROMPT_MARKER_SYSTEM) :].lstrip()
    return system.strip(), user.strip()
