#!/usr/bin/env bash
# Minimal SVO-AQA evaluation example.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/DeepASMR-NSpeech-dataset}"
BANK="${BANK:-${DATASET_ROOT}/SVO-AQA/test_subset/bank.json}"
PRED="${PRED:-${1:-predictions.json}}"

if [[ ! -f "$BANK" ]]; then
  echo "[ERROR] Missing bank: $BANK" >&2
  echo "Download: hf download AudioCC-Lab/DeepASMR-NSpeech --repo-type dataset --local-dir ${DATASET_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${ROOT}/benchmark/lib:${ROOT}/benchmark/svo_aqa/scripts:${PYTHONPATH:-}"
python3 "${ROOT}/benchmark/svo_aqa/scripts/eval_mc.py" \
  --bank "$BANK" \
  --predictions "$PRED"
