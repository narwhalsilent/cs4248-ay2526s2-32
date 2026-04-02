#!/bin/bash
#SBATCH --job-name=desarcastic_gen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

cd ~/cs4248-ay2526s2-32

mkdir -p logs
mkdir -p checkpoints

# module load python/3.10
# module load cuda/12.1

source .venv/bin/activate
export PATH="$HOME/cs4248-ay2526s2-32/.venv/bin:$PATH"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

SARCASM_MODEL_REPO="helinivan/english-sarcasm-detector"
TEACHER_MODEL_REPO="meta-llama/llama-3.1-8b-instruct"
INPUT_FILE="data/raw/Sarcasm_Headlines_Dataset.json"
OUTPUT_FILE="data/silver/desarcastic_headlines_1788.txt"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Error: OPENROUTER_API_KEY is not set."
    exit 1
fi

echo "Checking Hugging Face model cache for ${SARCASM_MODEL_REPO}..."
if ! python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "helinivan/english-sarcasm-detector"
try:
    snapshot_download(repo_id=repo_id, local_files_only=True)
    print(f"Found cached model: {repo_id}")
except Exception:
    raise SystemExit(1)
PY
then
    echo "Sarcasm model not found locally. Downloading from Hugging Face..."
    python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "helinivan/english-sarcasm-detector"
path = snapshot_download(repo_id=repo_id)
print(f"Downloaded {repo_id} to {path}")
PY
fi

python data/silver/generate_desarcastic_scripts.py \
    --input "${INPUT_FILE}" \
    --output "${OUTPUT_FILE}" \
    --model "${TEACHER_MODEL_REPO}"
