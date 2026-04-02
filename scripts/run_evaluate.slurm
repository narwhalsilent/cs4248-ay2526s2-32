#!/bin/bash
#SBATCH --job-name=satire_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

cd ~/cs4248-ay2526s2-32

mkdir -p logs
mkdir -p outputs/evaluation

source .venv/bin/activate
export PATH="$HOME/cs4248-ay2526s2-32/.venv/bin:$PATH"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

MODEL_PATH="${MODEL_PATH:-checkpoints/bart_satire_rebalanced/final}"
TEST_DATA="${TEST_DATA:-data/silver/rebalanced/test.csv}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
CANDIDATE_STRATEGY="${CANDIDATE_STRATEGY:-sample}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
STYLE_WEIGHT="${STYLE_WEIGHT:-2.0}"
SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-1.0}"
COPY_WEIGHT="${COPY_WEIGHT:-1.5}"
FLUENCY_WEIGHT="${FLUENCY_WEIGHT:-0.15}"
SAVE_PREFERENCE_PAIRS="${SAVE_PREFERENCE_PAIRS:-1}"
PREFERENCE_OUTPUT_JSONL="${PREFERENCE_OUTPUT_JSONL:-}"
PREFERENCE_PAIR_MODE="${PREFERENCE_PAIR_MODE:-best_vs_worst}"
PREFERENCE_MIN_MARGIN="${PREFERENCE_MIN_MARGIN:-0.1}"
NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
SOURCE_PREFIX="${SOURCE_PREFIX:-}"

echo "Evaluating model: ${MODEL_PATH}"
echo "Test data      : ${TEST_DATA}"
echo "Candidates     : ${NUM_CANDIDATES} (${CANDIDATE_STRATEGY})"
echo "Rerank weights : style=${STYLE_WEIGHT}, similarity=${SIMILARITY_WEIGHT}, copy=${COPY_WEIGHT}, fluency=${FLUENCY_WEIGHT}"

CMD=(
  python src/03_evaluation/evaluate.py
  --model_path "${MODEL_PATH}"
  --test_data "${TEST_DATA}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --num_beams "${NUM_BEAMS}"
  --num_candidates "${NUM_CANDIDATES}"
  --candidate_strategy "${CANDIDATE_STRATEGY}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --style_weight "${STYLE_WEIGHT}"
  --similarity_weight "${SIMILARITY_WEIGHT}"
  --copy_weight "${COPY_WEIGHT}"
  --fluency_weight "${FLUENCY_WEIGHT}"
  --num_examples "${NUM_EXAMPLES}"
)

if [ -n "${OUTPUT_CSV}" ]; then
  CMD+=(--output_csv "${OUTPUT_CSV}")
fi

if [ "${SAVE_PREFERENCE_PAIRS}" = "1" ]; then
  CMD+=(
    --save_preference_pairs
    --preference_pair_mode "${PREFERENCE_PAIR_MODE}"
    --preference_min_margin "${PREFERENCE_MIN_MARGIN}"
  )
fi

if [ -n "${PREFERENCE_OUTPUT_JSONL}" ]; then
  CMD+=(--preference_output_jsonl "${PREFERENCE_OUTPUT_JSONL}")
fi

if [ -n "${SOURCE_PREFIX}" ]; then
  CMD+=(--source_prefix "${SOURCE_PREFIX}")
fi

"${CMD[@]}"
