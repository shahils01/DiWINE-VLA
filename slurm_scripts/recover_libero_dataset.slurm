#!/bin/bash
#SBATCH --job-name=libero-recover
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/shahils/DiWINE-VLA-rebuild-20260906/recovery_logs/libero-recover-%j.out
#SBATCH --error=/scratch/shahils/DiWINE-VLA-rebuild-20260906/recovery_logs/libero-recover-%j.err

set -euo pipefail

export HF_HUB_DISABLE_XET=1

DATASET_DIR=/scratch/shahils/openpi/datasets/Libero-XVLA-format
HF_CACHE_DIR=/scratch/shahils/.cache/huggingface-rebuild
HF_CLI=/home/shahils/.local/bin/huggingface-cli

mkdir -p "$DATASET_DIR" "$HF_CACHE_DIR"

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"

# Palmetto's shared egress can exhaust Hugging Face's five-minute request window.
# Existing files resume automatically, so wait for the window to reset and use
# conservative concurrency on retries.
sleep 330

"$HF_CLI" download 2toINF/Libero-XVLA-format \
  --repo-type dataset \
  --cache-dir "$HF_CACHE_DIR" \
  --local-dir "$DATASET_DIR" \
  --max-workers 2

test -s "$DATASET_DIR/libero_meta.json"

HDF5_COUNT=$(find "$DATASET_DIR" -type f -name '*.hdf5' | wc -l)
TOTAL_BYTES=$(du -sb "$DATASET_DIR" | awk '{print $1}')

echo "HDF5 files: $HDF5_COUNT"
echo "Dataset bytes: $TOTAL_BYTES"

if [ "$HDF5_COUNT" -lt 5000 ]; then
  echo "Dataset validation failed: expected at least 5000 HDF5 files" >&2
  exit 1
fi

echo "Completed: $(date --iso-8601=seconds)"
