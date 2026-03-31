#!/bin/bash
#SBATCH --job-name=satire_train_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

cd ~/cs4248-ay2526s2-32

mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs/evaluation

source .venv/bin/activate
export PATH="$HOME/cs4248-ay2526s2-32/.venv/bin:$PATH"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

TARGET="${TARGET:-bart}"
CONFIG_PATH="${CONFIG_PATH:-}"
TEST_DATA="${TEST_DATA:-data/silver/rebalanced/test.csv}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
RUN_NAME="${RUN_NAME:-job${SLURM_JOB_ID:-manual}}"

if [ -n "${CONFIG_PATH}" ]; then
  CONFIGS=("${CONFIG_PATH}")
else
  case "${TARGET}" in
    bart)
      CONFIGS=("configs/bart_training_rebalanced.yaml")
      ;;
    t5)
      CONFIGS=("configs/t5_training_rebalanced.yaml")
      ;;
    both)
      CONFIGS=(
        "configs/bart_training_rebalanced.yaml"
        "configs/t5_training_rebalanced.yaml"
      )
      ;;
    *)
      echo "Error: TARGET must be one of: bart, t5, both"
      exit 1
      ;;
  esac
fi

for cfg_path in "${CONFIGS[@]}"; do
  CONFIG_NAME="$(basename "${cfg_path}" .yaml)"
  RUN_NAME_FOR_CFG="${RUN_NAME}_${CONFIG_NAME}"

  BASE_OUTPUT_DIR="$(
  python - <<'PY' "${cfg_path}"
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["training"]["output_dir"])
PY
  )"

  SOURCE_PREFIX="$(
  python - <<'PY' "${cfg_path}"
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["data"].get("source_prefix", ""))
PY
  )"

  FINAL_MODEL_PATH="${BASE_OUTPUT_DIR}_${RUN_NAME_FOR_CFG}/final"

  if [ -n "${OUTPUT_CSV:-}" ] && [ "${#CONFIGS[@]}" -eq 1 ]; then
    EVAL_OUTPUT_CSV="${OUTPUT_CSV}"
  else
    MODEL_DIR_NAME="$(basename "$(dirname "${FINAL_MODEL_PATH}")")"
    EVAL_OUTPUT_CSV="outputs/evaluation/${MODEL_DIR_NAME}/test_predictions.csv"
  fi

  echo "Training config : ${cfg_path}"
  echo "Run name        : ${RUN_NAME_FOR_CFG}"
  echo "Final model path: ${FINAL_MODEL_PATH}"
  echo "Test data       : ${TEST_DATA}"

  python src/02_training/train.py \
    --config "${cfg_path}" \
    --run_name "${RUN_NAME_FOR_CFG}"

  EVAL_CMD=(
    python src/03_evaluation/evaluate.py
    --model_path "${FINAL_MODEL_PATH}"
    --test_data "${TEST_DATA}"
    --batch_size "${BATCH_SIZE}"
    --max_length "${MAX_LENGTH}"
    --num_beams "${NUM_BEAMS}"
    --num_examples "${NUM_EXAMPLES}"
    --output_csv "${EVAL_OUTPUT_CSV}"
  )

  if [ -n "${SOURCE_PREFIX}" ]; then
    EVAL_CMD+=(--source_prefix "${SOURCE_PREFIX}")
  fi

  "${EVAL_CMD[@]}"
done
