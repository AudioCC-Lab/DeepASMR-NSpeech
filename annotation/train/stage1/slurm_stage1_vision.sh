#!/bin/bash
# Stage1 视觉：train_events.json 每 event 一张 fig_train 关键帧 → visual_description
# 脚本与 prompt 均在 train/stage1/ 下，便于复现。
#
# 本地试跑 10 条：
#   cd /path/to/DeepASMR-NSpeech
#   python -u run_stage1_vision.py \
#     --limit 10 --no-enable-thinking --max-tokens 1024 --continue-on-error
#
# 提交全量：
#   mkdir -p annotation/train/stage1/logs
#   sbatch annotation/train/stage1/slurm_stage1_vision.sh
#
# 试跑 10 条（Slurm）：
#   sbatch --export=ALL,STAGE1_EXTRA='--limit 10 --no-enable-thinking --max-tokens 1024' \
#     slurm_stage1_vision.sh
#
# API / 模型：${ROOT}/.env（SILICONFLOW_API_KEY、INFO_FULL_VERIFY_VISION_MODEL）
#
#SBATCH --job-name=train_s1_vis
#SBATCH --output=annotation/train/stage1/logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --qos=qnormal
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL

ROOT="${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIB_DIR="${LIB_DIR:-${ROOT}/annotation/train/stage2/lib}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT}/annotation/train/stage1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_JSON="${INPUT_JSON:-${ROOT}/annotation/train/train_events.json}"
OUT_JSON="${OUT_JSON:-${ROOT}/annotation/train/train_events_with_visual.json}"
FIGS_DIR="${FIGS_DIR:-${ROOT}/annotation/train/frames}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-${STAGE1_DIR}/prompts/stage1_system.txt}"
USER_PROMPT="${USER_PROMPT:-${STAGE1_DIR}/prompts/stage1_user.txt}"

RUN_STEMS="${RUN_STEMS:-}"
STAGE1_EXTRA="${STAGE1_EXTRA:---no-enable-thinking --max-tokens 1024 --continue-on-error}"

set -euo pipefail
mkdir -p "${STAGE1_DIR}/logs"

echo "[$(date)] JOBID=${SLURM_JOB_ID:-local}"
echo "  ROOT=${ROOT}"
echo "  STAGE1_DIR=${STAGE1_DIR}"
echo "  INPUT_JSON=${INPUT_JSON}"
echo "  OUT_JSON=${OUT_JSON}"
echo "  FIGS_DIR=${FIGS_DIR}"
echo "  SYSTEM_PROMPT=${SYSTEM_PROMPT}"
echo "  USER_PROMPT=${USER_PROMPT}"
if [[ -n "${RUN_STEMS}" ]]; then
  echo "  RUN_STEMS=${RUN_STEMS}"
fi
if [[ -n "${STAGE1_EXTRA}" ]]; then
  echo "  STAGE1_EXTRA=${STAGE1_EXTRA}"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${LIB_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"

CMD=("${PYTHON_BIN}" -u "${STAGE1_DIR}/run_stage1_vision.py"
  --source-json "${INPUT_JSON}"
  --out-json "${OUT_JSON}"
  --figs-dir "${FIGS_DIR}"
  --system-prompt-file "${SYSTEM_PROMPT}"
  --user-prompt-file "${USER_PROMPT}")

if [[ -n "${RUN_STEMS}" ]]; then
  CMD+=(--stems "${RUN_STEMS}")
fi

if [[ -n "${STAGE1_EXTRA}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${STAGE1_EXTRA})
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "[$(date)] ${CMD[*]}"
"${CMD[@]}"

# 与 train_events.json 一致：去掉 preview/intro/combination/medley-mix
FILTER_LOG="${STAGE1_DIR}/logs/filter_blocked_${SLURM_JOB_ID:-local}.json"
"${PYTHON_BIN}" -u "${STAGE1_DIR}/filter_blocked_train_events.py" \
  "${OUT_JSON}" \
  --log "${FILTER_LOG}"

echo "[$(date)] Job finished."
