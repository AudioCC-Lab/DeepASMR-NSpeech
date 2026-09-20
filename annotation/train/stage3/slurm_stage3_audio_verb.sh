#!/bin/bash
# Stage3：Omni 音频模型仅选动词（名词/class 来自 stage2）
# 公共运行库与词表位于仓库根目录；本 stage 不做 GT 评测。
#
# 本地试跑:
#   cd /path/to/DeepASMR-NSpeech
#   python -u run_stage3_audio_verb.py --dry-run --limit 3
#
# 提交全量:
#   mkdir -p annotation/train/stage3/logs
#   sbatch annotation/train/stage3/slurm_stage3_audio_verb.sh
#
# 试跑 10 条（Slurm）:
#   sbatch --export=ALL,STAGE3_EXTRA='--limit 10 --continue-on-error --checkpoint-every 5' \
#     slurm_stage3_audio_verb.sh
#
# 超时后重新 sbatch 即可断点续跑（已有 verb 的 event 会跳过）
# API：stage3/.env 或项目根 .env（DASHSCOPE_API_KEY、DASHSCOPE_OMNI_MODEL）
#
#SBATCH --job-name=train_s3_omni
#SBATCH --output=annotation/train/stage3/logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --qos=qnormal
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL

ROOT="${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STAGE3_DIR="${STAGE3_DIR:-${ROOT}/annotation/train/stage3}"
OUT_JSON="${OUT_JSON:-${ROOT}/annotation/train/train_events_with_visual.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROMPT="${PROMPT:-${STAGE3_DIR}/prompts/audio_verb_only_with_text.txt}"
VERB_CSV="${VERB_CSV:-${ROOT}/vocab/asmr_verb_classes_en.csv}"
DISAMBIG="${DISAMBIG:-${ROOT}/vocab/stage2_class_disambiguation.txt}"
RUN_STEMS="${RUN_STEMS:-}"
STAGE3_EXTRA="${STAGE3_EXTRA:---continue-on-error --checkpoint-every 5}"

set -euo pipefail
mkdir -p "${STAGE3_DIR}/logs"

echo "[$(date)] JOBID=${SLURM_JOB_ID:-local}"
echo "  ROOT=${ROOT}"
echo "  STAGE3_DIR=${STAGE3_DIR}"
echo "  OUT_JSON=${OUT_JSON}"
echo "  PROMPT=${PROMPT}"
echo "  VERB_CSV=${VERB_CSV}"
echo "  DISAMBIG=${DISAMBIG}"
if [[ -n "${RUN_STEMS}" ]]; then
  echo "  RUN_STEMS=${RUN_STEMS}"
fi
if [[ -n "${STAGE3_EXTRA}" ]]; then
  echo "  STAGE3_EXTRA=${STAGE3_EXTRA}"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${STAGE3_DIR}/lib:${STAGE3_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"

CMD=("${PYTHON_BIN}" -u "${STAGE3_DIR}/run_stage3_audio_verb.py"
  --out-json "${OUT_JSON}"
  --prompt "${PROMPT}"
  --verb-classes-csv "${VERB_CSV}"
  --disambiguation-txt "${DISAMBIG}")

if [[ -n "${RUN_STEMS}" ]]; then
  CMD+=(--stems "${RUN_STEMS}")
fi

if [[ -n "${STAGE3_EXTRA}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${STAGE3_EXTRA})
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "[$(date)] ${CMD[*]}"
"${CMD[@]}"
echo "[$(date)] Job finished."
