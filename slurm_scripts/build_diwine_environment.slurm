#!/bin/bash
#SBATCH --job-name=diwine-env
#SBATCH --account=cuuser_yue6_imitation_learning_shahil
#SBATCH --partition=work1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/shahils/DiWINE-VLA-rebuild-20260906/recovery_logs/diwine-env-%j.out
#SBATCH --error=/scratch/shahils/DiWINE-VLA-rebuild-20260906/recovery_logs/diwine-env-%j.err

set -euo pipefail

CONDA_BIN=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/anaconda3-2023.09-0-3mhml42fa64byxqyd5fig5tbih625dp2/bin/conda
ENV_DIR=/scratch/shahils/conda_envs/diwine-vla-rebuild
REPO_DIR=/scratch/shahils/DiWINE-VLA-rebuild-20260906
PIP_CACHE_DIR=/scratch/shahils/.cache/pip-rebuild

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"

if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$CONDA_BIN" create -y -p "$ENV_DIR" python=3.10 pip
fi

"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
"$ENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements.txt"
"$ENV_DIR/bin/python" -m pip install wandb

"$ENV_DIR/bin/python" -c "import torch, transformers, accelerate, h5py, peft, safetensors, numpy; print('torch', torch.__version__); print('cuda build', torch.version.cuda); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('h5py', h5py.__version__); print('peft', peft.__version__); print('numpy', numpy.__version__)"

echo "Completed: $(date --iso-8601=seconds)"
