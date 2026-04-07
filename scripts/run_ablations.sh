#!/bin/bash
# run_ablations.sh

export PYTHONUNBUFFERED=1

# Exit immediately if a command exits with a non-zero status
set -e

# Define your parameter space
CONFIGS=(
  "configs/bart_training_full.yaml" 
  "configs/bart_training_truncated.yaml" 
  "configs/t5_training_full.yaml" 
  "configs/t5_training_truncated.yaml"
)
STRATEGIES=("vanilla" "weighted" "curriculum")

echo "Starting SFT Ablation Sweep on 2x T4 GPUs..."

for config in "${CONFIGS[@]}"; do
  # Safety check: skip if the config file wasn't uploaded correctly
  if [ ! -f "$config" ]; then
    echo "Warning: $config not found, skipping..."
    continue
  fi

  for strategy in "${STRATEGIES[@]}"; do
    # Create a dynamic run name (e.g., bart_training_full_weighted)
    base_config=$(basename "$config" .yaml)
    run_name="${base_config}_${strategy}"

    echo "================================================================="
    echo "Launching: $run_name"
    echo "   Config: $config | Strategy: $strategy"
    echo "================================================================="

    # Execute using torchrun for multi-GPU
    torchrun --nproc_per_node=2 train.py \
      --config "$config" \
      --run_name "$run_name" \
      --training_strategy "$strategy"

    echo "Completed: $run_name"
    echo "-----------------------------------------------------------------"
  done
done

echo "All ablation runs successfully finished!"