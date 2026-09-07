#!/bin/bash
#SBATCH --job-name=diwine-smoke
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shahils/DiWINE-VLA-rebuild-20260906/recovery_logs/diwine-smoke-%j.out
#SBATCH --error=/scratch/shahils/DiWINE-VLA-rebuild-20260906/recovery_logs/diwine-smoke-%j.err

set -euo pipefail

REPO_DIR=/scratch/shahils/DiWINE-VLA-rebuild-20260906
ENV_DIR=/scratch/shahils/conda_envs/diwine-vla-rebuild
DATASET_META=/scratch/shahils/openpi/datasets/Libero-XVLA-format/libero_meta.json
OUTPUT_ROOT=/scratch/shahils/diwine_runs/baseline_smoke

export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_CACHE=/scratch/shahils/.cache/huggingface-hub-rebuild
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="$OUTPUT_ROOT/job-$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR" "$HF_HUB_CACHE"

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Output: $OUTPUT_DIR"
nvidia-smi

test -s "$DATASET_META"
cd "$REPO_DIR"

"$ENV_DIR/bin/accelerate" launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train.py \
  --models 2toINF/X-VLA-Pt \
  --train_metas_path "$DATASET_META" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 1 \
  --num_workers 0 \
  --num_views 3 \
  --learning_rate 1e-4 \
  --iters 2 \
  --freeze_steps 1 \
  --warmup_steps 1 \
  --save_interval 2 \
  --log_interval 1 \
  --use_future_prediction \
  --future_offset_steps 1 \
  --future_horizon_steps 2 \
  --future_latent_dim 1024 \
  --future_hidden_dim 1024 \
  --future_nll_weight 0.5 \
  --future_kl_weight 0.2 \
  --num_future_samples 2

test -s "$OUTPUT_DIR/ckpt-2/model.safetensors"
test -s "$OUTPUT_DIR/ckpt-2/state.json"

echo "Checkpoint bytes: $(du -sb "$OUTPUT_DIR/ckpt-2" | awk '{print $1}')"
echo "Completed: $(date --iso-8601=seconds)"
