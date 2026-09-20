#!/usr/bin/env bash
set -euo pipefail

# Upload a prepared DeepASMR-NSpeech dataset directory. Authentication is read
# by the Hugging Face CLI from its credential store or HF_TOKEN. Never place a
# token in this script.

DATASET_DIR="${DATASET_DIR:-./DeepASMR-NSpeech-dataset}"
HF_REPO="${HF_REPO:-AudioCC-Lab/DeepASMR-NSpeech}"

if [[ ! -f "${DATASET_DIR}/release_manifest.json" ]]; then
  echo "[ERROR] Missing ${DATASET_DIR}/release_manifest.json" >&2
  echo "Run scripts/prepare_deepasmr_nspeech_hf.py first." >&2
  exit 1
fi

command -v hf >/dev/null || {
  echo "[ERROR] Hugging Face CLI 'hf' is not installed." >&2
  exit 1
}

echo "[INFO] Uploading ${DATASET_DIR} to dataset ${HF_REPO}"
python3 - "$HF_REPO" <<'PY'
import sys
from huggingface_hub import HfApi

HfApi().create_repo(sys.argv[1], repo_type="dataset", private=False, exist_ok=True)
PY
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
hf upload "${HF_REPO}" "${DATASET_DIR}" . --repo-type dataset
