#!/bin/bash
#SBATCH --job-name=satire_compare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=18:00:00

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
BASELINES="${BASELINES:-sft,sft_rerank,dpo,dpo_rerank}"
TEST_DATA="${TEST_DATA:-data/silver/rebalanced/test.csv}"
RUN_NAME="${RUN_NAME:-job${SLURM_JOB_ID:-manual}}"

BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NUM_EXAMPLES="${NUM_EXAMPLES:-6}"

RERANK_NUM_CANDIDATES="${RERANK_NUM_CANDIDATES:-10}"
RERANK_CANDIDATE_STRATEGY="${RERANK_CANDIDATE_STRATEGY:-sample}"
RERANK_TEMPERATURE="${RERANK_TEMPERATURE:-0.9}"
RERANK_TOP_P="${RERANK_TOP_P:-0.95}"
RERANK_STYLE_WEIGHT="${RERANK_STYLE_WEIGHT:-1.0}"
RERANK_SIMILARITY_WEIGHT="${RERANK_SIMILARITY_WEIGHT:-1.5}"
RERANK_COPY_WEIGHT="${RERANK_COPY_WEIGHT:-1.0}"
RERANK_FLUENCY_WEIGHT="${RERANK_FLUENCY_WEIGHT:-0.15}"

PREFERENCE_NUM_CANDIDATES="${PREFERENCE_NUM_CANDIDATES:-8}"
PREFERENCE_CANDIDATE_STRATEGY="${PREFERENCE_CANDIDATE_STRATEGY:-sample}"
PREFERENCE_TEMPERATURE="${PREFERENCE_TEMPERATURE:-0.9}"
PREFERENCE_TOP_P="${PREFERENCE_TOP_P:-0.95}"
PREFERENCE_STYLE_WEIGHT="${PREFERENCE_STYLE_WEIGHT:-1.0}"
PREFERENCE_SIMILARITY_WEIGHT="${PREFERENCE_SIMILARITY_WEIGHT:-1.5}"
PREFERENCE_COPY_WEIGHT="${PREFERENCE_COPY_WEIGHT:-1.0}"
PREFERENCE_FLUENCY_WEIGHT="${PREFERENCE_FLUENCY_WEIGHT:-0.15}"
PREFERENCE_PAIR_MODE="${PREFERENCE_PAIR_MODE:-best_vs_worst}"
PREFERENCE_MIN_MARGIN="${PREFERENCE_MIN_MARGIN:-0.05}"

DPO_NUM_EPOCHS="${DPO_NUM_EPOCHS:-3}"
DPO_TRAIN_BATCH_SIZE="${DPO_TRAIN_BATCH_SIZE:-8}"
DPO_EVAL_BATCH_SIZE="${DPO_EVAL_BATCH_SIZE:-8}"
DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-1e-5}"
DPO_WEIGHT_DECAY="${DPO_WEIGHT_DECAY:-0.01}"
DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.05}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-1}"
DPO_MAX_GRAD_NORM="${DPO_MAX_GRAD_NORM:-1.0}"
DPO_CHECKPOINT_STAGE="${DPO_CHECKPOINT_STAGE:-best}"

contains_baseline() {
  case ",${BASELINES}," in
    *,"$1",*) return 0 ;;
    *) return 1 ;;
  esac
}

print_summary_table() {
  local summary_csv="$1"
  python - <<'PY' "${summary_csv}"
import csv
import os
import sys

path = sys.argv[1]
if not os.path.exists(path):
    print(f"Summary file not found: {path}")
    raise SystemExit(0)

with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if not rows:
    print("No rows in summary file.")
    raise SystemExit(0)

columns = [
    ("baseline", 16),
    ("style_accuracy", 9),
    ("content_preservation", 9),
    ("fluency_perplexity", 9),
    ("exact_copy_rate", 9),
    ("near_copy_rate", 9),
]

header = "  ".join(name.ljust(width) for name, width in columns)
print(header)
print("-" * len(header))
for row in rows:
    formatted = []
    for name, width in columns:
        value = row[name]
        if name == "baseline":
            formatted.append(value.ljust(width))
        else:
            formatted.append(f"{float(value):>{width}.4f}")
    print("  ".join(formatted))
PY
}

append_summary_row() {
  local summary_csv="$1"
  local baseline="$2"
  local log_path="$3"
  python - <<'PY' "${summary_csv}" "${baseline}" "${log_path}"
import csv
import os
import re
import sys

summary_csv, baseline, log_path = sys.argv[1:4]
patterns = {
    "style_accuracy": r"Style Accuracy \(RoBERTa sarcasm prob\)\s*:\s*([0-9.]+)",
    "content_preservation": r"Content Preservation \(SBERT cosine\)\s*:\s*([0-9.]+)",
    "fluency_perplexity": r"Linguistic Fluency \(GPT-2 perplexity\)\s*:\s*([0-9.]+)",
    "exact_copy_rate": r"Exact Copy Rate\s*:\s*([0-9.]+)",
    "near_copy_rate": r"Near-Copy Rate \(sim >= 0.90\)\s*:\s*([0-9.]+)",
}

with open(log_path, encoding="utf-8") as handle:
    text = handle.read()

row = {"baseline": baseline}
for key, pattern in patterns.items():
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not find metric '{key}' in {log_path}")
    row[key] = match.group(1)

fieldnames = [
    "baseline",
    "style_accuracy",
    "content_preservation",
    "fluency_perplexity",
    "exact_copy_rate",
    "near_copy_rate",
]
write_header = not os.path.exists(summary_csv)
with open(summary_csv, "a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

run_eval() {
  local baseline="$1"
  local model_path="$2"
  local summary_csv="$3"
  local output_dir="$4"
  local source_prefix="$5"
  local use_rerank="$6"

  local eval_log="${output_dir}/${baseline}.log"
  local eval_csv="${output_dir}/${baseline}_predictions.csv"

  local cmd=(
    python src/03_evaluation/evaluate.py
    --model_path "${model_path}"
    --test_data "${TEST_DATA}"
    --batch_size "${BATCH_SIZE}"
    --max_length "${MAX_LENGTH}"
    --num_beams "${NUM_BEAMS}"
    --num_examples "${NUM_EXAMPLES}"
    --output_csv "${eval_csv}"
  )

  if [ "${use_rerank}" = "1" ]; then
    cmd+=(
      --num_candidates "${RERANK_NUM_CANDIDATES}"
      --candidate_strategy "${RERANK_CANDIDATE_STRATEGY}"
      --temperature "${RERANK_TEMPERATURE}"
      --top_p "${RERANK_TOP_P}"
      --style_weight "${RERANK_STYLE_WEIGHT}"
      --similarity_weight "${RERANK_SIMILARITY_WEIGHT}"
      --copy_weight "${RERANK_COPY_WEIGHT}"
      --fluency_weight "${RERANK_FLUENCY_WEIGHT}"
    )
  else
    cmd+=(--num_candidates 1)
  fi

  if [ -n "${source_prefix}" ]; then
    cmd+=(--source_prefix "${source_prefix}")
  fi

  echo
  echo "================================================================"
  echo "Baseline      : ${baseline}"
  echo "Model path    : ${model_path}"
  echo "Reranking     : ${use_rerank}"
  echo "Output CSV    : ${eval_csv}"
  echo "Log file      : ${eval_log}"
  echo "================================================================"

  "${cmd[@]}" 2>&1 | tee "${eval_log}"
  append_summary_row "${summary_csv}" "${baseline}" "${eval_log}"
}

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
  CONFIG_TAG="$(
  python - <<'PY' "${cfg_path}"
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["model"]["type"])
PY
  )"

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

  if [ "${#CONFIGS[@]}" -eq 1 ]; then
    RUN_NAME_FOR_CFG="${RUN_NAME}"
  else
    RUN_NAME_FOR_CFG="${RUN_NAME}_${CONFIG_TAG}"
  fi

  SFT_MODEL_DIR="${BASE_OUTPUT_DIR}_${RUN_NAME_FOR_CFG}"
  SFT_MODEL_PATH="${SFT_MODEL_DIR}/final"
  PREF_DIR="${SFT_MODEL_DIR}/preferences"
  DPO_OUTPUT_DIR="${SFT_MODEL_DIR}_dpo"
  DPO_MODEL_PATH="${DPO_OUTPUT_DIR}/${DPO_CHECKPOINT_STAGE}"
  SUMMARY_DIR="outputs/evaluation/$(basename "${SFT_MODEL_DIR}")"
  SUMMARY_CSV="${SUMMARY_DIR}/baseline_summary.csv"

  mkdir -p "${SUMMARY_DIR}"
  rm -f "${SUMMARY_CSV}"

  echo
  echo "############################################################"
  echo "Config         : ${cfg_path}"
  echo "Run name       : ${RUN_NAME_FOR_CFG}"
  echo "Baselines      : ${BASELINES}"
  echo "SFT model dir  : ${SFT_MODEL_DIR}"
  echo "DPO model dir  : ${DPO_OUTPUT_DIR}"
  echo "Summary CSV    : ${SUMMARY_CSV}"
  echo "############################################################"

  NEED_SFT=0
  NEED_DPO=0
  if contains_baseline sft || contains_baseline sft_rerank || contains_baseline dpo || contains_baseline dpo_rerank; then
    NEED_SFT=1
  fi
  if contains_baseline dpo || contains_baseline dpo_rerank; then
    NEED_DPO=1
  fi

  if [ "${NEED_SFT}" = "1" ]; then
    TRAIN_CMD=(
      python src/02_training/train.py
      --config "${cfg_path}"
      --run_name "${RUN_NAME_FOR_CFG}"
    )

    if [ "${NEED_DPO}" = "1" ]; then
      TRAIN_CMD+=(
        --export_preferences
        --preference_output_dir "${PREF_DIR}"
        --preference_num_candidates "${PREFERENCE_NUM_CANDIDATES}"
        --preference_candidate_strategy "${PREFERENCE_CANDIDATE_STRATEGY}"
        --preference_temperature "${PREFERENCE_TEMPERATURE}"
        --preference_top_p "${PREFERENCE_TOP_P}"
        --preference_style_weight "${PREFERENCE_STYLE_WEIGHT}"
        --preference_similarity_weight "${PREFERENCE_SIMILARITY_WEIGHT}"
        --preference_copy_weight "${PREFERENCE_COPY_WEIGHT}"
        --preference_fluency_weight "${PREFERENCE_FLUENCY_WEIGHT}"
        --preference_pair_mode "${PREFERENCE_PAIR_MODE}"
        --preference_min_margin "${PREFERENCE_MIN_MARGIN}"
      )
    fi

    echo
    echo "Starting SFT training..."
    "${TRAIN_CMD[@]}"
  fi

  if [ "${NEED_DPO}" = "1" ]; then
    echo
    echo "Starting DPO training..."
    python src/02_training/train_dpo.py \
      --model_path "${SFT_MODEL_PATH}" \
      --train_preferences "${PREF_DIR}/train_preference_pairs.jsonl" \
      --validation_preferences "${PREF_DIR}/validation_preference_pairs.jsonl" \
      --output_dir "${DPO_OUTPUT_DIR}" \
      --max_source_length "${MAX_LENGTH}" \
      --max_target_length "${MAX_LENGTH}" \
      --per_device_train_batch_size "${DPO_TRAIN_BATCH_SIZE}" \
      --per_device_eval_batch_size "${DPO_EVAL_BATCH_SIZE}" \
      --gradient_accumulation_steps "${DPO_GRAD_ACCUM}" \
      --num_train_epochs "${DPO_NUM_EPOCHS}" \
      --learning_rate "${DPO_LEARNING_RATE}" \
      --weight_decay "${DPO_WEIGHT_DECAY}" \
      --warmup_ratio "${DPO_WARMUP_RATIO}" \
      --max_grad_norm "${DPO_MAX_GRAD_NORM}" \
      --beta "${DPO_BETA}"
  fi

  if contains_baseline sft; then
    run_eval "sft" "${SFT_MODEL_PATH}" "${SUMMARY_CSV}" "${SUMMARY_DIR}" "${SOURCE_PREFIX}" 0
  fi
  if contains_baseline sft_rerank; then
    run_eval "sft_rerank" "${SFT_MODEL_PATH}" "${SUMMARY_CSV}" "${SUMMARY_DIR}" "${SOURCE_PREFIX}" 1
  fi
  if contains_baseline dpo; then
    run_eval "dpo" "${DPO_MODEL_PATH}" "${SUMMARY_CSV}" "${SUMMARY_DIR}" "${SOURCE_PREFIX}" 0
  fi
  if contains_baseline dpo_rerank; then
    run_eval "dpo_rerank" "${DPO_MODEL_PATH}" "${SUMMARY_CSV}" "${SUMMARY_DIR}" "${SOURCE_PREFIX}" 1
  fi

  echo
  echo "================ Baseline Summary: ${RUN_NAME_FOR_CFG} ================"
  print_summary_table "${SUMMARY_CSV}"
  echo "Summary CSV saved to: ${SUMMARY_CSV}"
done
