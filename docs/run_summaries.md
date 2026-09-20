# Operational run summaries

Long-running network, media, and API stages use `deepasmr_nspeech.RunSummary`.
A summary is written even when `--continue-on-error` allows partial completion.

```json
{
  "stage": "annotation.train.stage3_audio_verb",
  "total": 100,
  "counts": {"success": 91, "failed": 3, "skipped": 6},
  "failure_reasons": {"missing_audio": 4, "rate_limit": 2, "already_completed": 2},
  "failures": [{"item_id": "example", "reason": "rate_limit"}]
}
```

Failure details are bounded so a large run does not produce an unbounded log.
The classifier recognizes rate limits, timeouts, network errors,
authentication, unavailable/too-long sources, media processing, storage,
invalid/missing inputs, and unknown errors.

These files are deliberately separate from label-quality measurement. A
successful count means the pipeline produced its expected record, not that a
private GT comparison passed.
