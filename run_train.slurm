#!/bin/bash
#SBATCH --job-name=satire_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

# Go to the copied project directory on the cluster
cd ~/cs4248-ay2526s2-32

# Create logs directory if it does not exist
mkdir -p logs

# Optional: create checkpoint parent dir too
mkdir -p checkpoints

# Load cluster modules as needed
module load python/3.10
# module load cuda/12.1

# Activate environment
source venv/bin/activate
# or: conda activate <your-env-name>

# Run training with an existing config
python src/02_training/train.py --config configs/bart_training.yaml
# For T5 instead:
# python src/02_training/train.py --config configs/t5_training.yaml
