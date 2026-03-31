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
NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
SOURCE_PREFIX="${SOURCE_PREFIX:-}"

echo "Evaluating model: ${MODEL_PATH}"
echo "Test data      : ${TEST_DATA}"

CMD=(
  python src/03_evaluation/evaluate.py
  --model_path "${MODEL_PATH}"
  --test_data "${TEST_DATA}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --num_beams "${NUM_BEAMS}"
  --num_examples "${NUM_EXAMPLES}"
)

if [ -n "${OUTPUT_CSV}" ]; then
  CMD+=(--output_csv "${OUTPUT_CSV}")
fi

if [ -n "${SOURCE_PREFIX}" ]; then
  CMD+=(--source_prefix "${SOURCE_PREFIX}")
fi

"${CMD[@]}"
