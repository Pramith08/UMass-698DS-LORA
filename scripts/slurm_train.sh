#!/usr/bin/env bash
#SBATCH --job-name=lora-judge
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --partition=gpu              # adjust for your Unity partition (gpu / gpu-long / etc.)
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00              # 4 hours; adjust if needed

# Run from code/LoRA/ directory.
# Adjust the module loads / venv activation for your Unity environment.

set -euo pipefail

# module load cuda/12.1
# module load python/3.11
# source ../../.venv/bin/activate

mkdir -p outputs

echo "==> Phase 1: data prep"
python prep_data.py

echo "==> Phase 3: LoRA training"
python train.py

echo "==> Phase 4: test-split eval"
python eval_test.py

echo "==> Done. See outputs/metrics_test.json"
