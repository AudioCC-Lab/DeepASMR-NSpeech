#!/bin/bash
# Stage2：文本 LLM → verb 大类 top-2 + subject + object
# 公共运行库与词表位于仓库根目录，提示词位于本 stage 目录。
#
# 本地试跑:
#   cd /path/to/DeepASMR-NSpeech
#   python -u run_stage2_text_class_noun.py --dry-run --limit 3
#
# 提交全量:
#   mkdir -p annotation/train/stage2/logs
#   sbatch annotation/train/stage2/slurm_stage2_text_class_noun.sh
#
# 试跑 10 条（Slurm）:
#   sbatch --export=ALL,STAGE2_EXTRA='--limit 10 --continue-on-error --checkpoint-every 10' \
#     slurm_stage2_text_class_noun.sh
#
# 超时后重新 sbatch 即可断点续跑（已有 asmr_class 的 event 会跳过）
# API：stage2/.env 或项目根 .env（SILICONFLOW_API_KEY、MODEL_TEXT）
#
#SBATCH --job-name=train_s2_txt
#SBATCH --output=annotation/train/stage2/logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --qos=qnormal
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL

ROOT="${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STAGE2_DIR="${STAGE2_DIR:-${ROOT}/annotation/train/stage2}"
OUT_JSON="${OUT_JSON:-${ROOT}/annotation/train/train_events_with_visual.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROMPT="${PROMPT:-${STAGE2_DIR}/prompts/class_and_noun_joint.txt}"
VERB_CSV="${VERB_CSV:-${ROOT}/vocab/asmr_verb_classes_en.csv}"
RUN_STEMS="${RUN_STEMS:-}"
STAGE2_EXTRA="${STAGE2_EXTRA:---continue-on-error --checkpoint-every 10}"

set -euo pipefail
mkdir -p "${STAGE2_DIR}/logs"

echo "[$(date)] JOBID=${SLURM_JOB_ID:-local}"
echo "  ROOT=${ROOT}"
echo "  STAGE2_DIR=${STAGE2_DIR}"
echo "  OUT_JSON=${OUT_JSON}"
echo "  PROMPT=${PROMPT}"
echo "  VERB_CSV=${VERB_CSV}"
if [[ -n "${RUN_STEMS}" ]]; then
  echo "  RUN_STEMS=${RUN_STEMS}"
fi
if [[ -n "${STAGE2_EXTRA}" ]]; then
  echo "  STAGE2_EXTRA=${STAGE2_EXTRA}"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${STAGE2_DIR}/lib:${STAGE2_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"

CMD=("${PYTHON_BIN}" -u "${STAGE2_DIR}/run_stage2_text_class_noun.py"
  --out-json "${OUT_JSON}"
  --prompt "${PROMPT}"
  --verb-classes-csv "${VERB_CSV}")

if [[ -n "${RUN_STEMS}" ]]; then
  CMD+=(--stems "${RUN_STEMS}")
fi

if [[ -n "${STAGE2_EXTRA}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${STAGE2_EXTRA})
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "[$(date)] ${CMD[*]}"
"${CMD[@]}"
echo "[$(date)] Job finished."
