"""Operational run summaries for collection and annotation stages.

The report deliberately measures pipeline execution rather than annotation
quality.  It does not require private human ground truth.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_FAILURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rate_limit", ("rate limit", "too many requests", "http 429", "status 429", "quota exceeded")),
    ("timeout", ("timed out", "timeout", "deadline exceeded")),
    (
        "network",
        (
            "connection reset",
            "connection refused",
            "connection aborted",
            "temporary failure in name resolution",
            "name or service not known",
            "network is unreachable",
            "remote end closed",
            "http 502",
            "http 503",
            "http 504",
        ),
    ),
    ("authentication", ("unauthorized", "forbidden", "http 401", "http 403", "sign in", "cookie")),
    (
        "source_unavailable",
        ("video unavailable", "private video", "removed", "not available", "members-only", "age-restricted"),
    ),
    ("source_too_long", ("video too long", "duration exceeds", "too long", "maximum duration")),
    ("media_processing", ("ffmpeg", "ffprobe", "decode", "invalid data found", "unsupported codec")),
    ("missing_input", ("no such file", "not found", "missing audio", "missing image", "不存在")),
    ("storage", ("no space left", "disk quota", "read-only file system")),
    ("invalid_input", ("invalid input", "invalid url", "parse error", "jsondecodeerror", "缺少有效")),
    ("api_error", ("api error", "internal server error", "service unavailable", "bad gateway")),
)


def _clean_detail(value: Any, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def classify_failure(error: Any) -> str:
    """Map an exception or message to a stable operational reason code."""

    text = _clean_detail(error).lower()
    for reason, needles in _FAILURE_PATTERNS:
        if any(needle in text for needle in needles):
            return reason
    return "unknown"


class RunSummary:
    """Accumulate success/skip/failure counts and write a JSON report."""

    def __init__(self, stage: str, *, max_failure_details: int = 200) -> None:
        self.stage = stage
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.status_counts: Counter[str] = Counter()
        self.reason_counts: Counter[str] = Counter()
        self.failures: list[dict[str, str]] = []
        self.max_failure_details = max(0, max_failure_details)
        self.dropped_failure_details = 0

    def add(
        self,
        status: str,
        *,
        item_id: Any = None,
        reason: str | None = None,
        detail: Any = None,
    ) -> None:
        status = status.strip().lower()
        self.status_counts[status] += 1
        if reason:
            self.reason_counts[reason] += 1
        if status in {"failed", "error"}:
            normalized_reason = reason or classify_failure(detail)
            if not reason:
                self.reason_counts[normalized_reason] += 1
            row = {"reason": normalized_reason}
            if item_id is not None:
                row["item_id"] = _clean_detail(item_id, limit=200)
            cleaned = _clean_detail(detail)
            if cleaned:
                row["detail"] = cleaned
            if len(self.failures) < self.max_failure_details:
                self.failures.append(row)
            else:
                self.dropped_failure_details += 1

    def payload(self, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        finished = datetime.now(timezone.utc)
        counts = dict(sorted(self.status_counts.items()))
        return {
            "stage": self.stage,
            "started_at": self.started_at.isoformat().replace("+00:00", "Z"),
            "finished_at": finished.isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 3),
            "total": sum(self.status_counts.values()),
            "counts": counts,
            "successful": counts.get("success", 0),
            "failed": counts.get("failed", 0) + counts.get("error", 0),
            "skipped": counts.get("skipped", 0),
            "failure_reasons": dict(sorted(self.reason_counts.items())),
            "failures": self.failures,
            "dropped_failure_details": self.dropped_failure_details,
            "metadata": metadata or {},
        }

    def finish(self, path: Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = self.payload(metadata=metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        counts = payload["counts"]
        print(
            f"[RUN-SUMMARY] stage={self.stage} total={payload['total']} "
            f"success={counts.get('success', 0)} skipped={counts.get('skipped', 0)} "
            f"failed={payload['failed']} reasons={payload['failure_reasons']} -> {path}"
        )
        return payload
