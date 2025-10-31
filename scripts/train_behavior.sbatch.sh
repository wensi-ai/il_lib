#!/bin/bash
#SBATCH --job-name="train_behavior"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --nodes=2
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/sc/train_behavior_%j.out
#SBATCH --error=outputs/sc/train_behavior_%j.err

set -euo pipefail

echo "SLURM_JOBID=${SLURM_JOBID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-}"
echo "working directory=${SLURM_SUBMIT_DIR:-$PWD}"

# ==== 环境 ====
source /vision/u/yinhang/miniconda3/bin/activate behavior
export KIT_DISABLE_PIP_PREBUNDLE=1
export OMNI_KIT_DISABLE_PIP_PREBUNDLE=1
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ==== 只用环境变量，不用 ~/.netrc ====
: "${WANDB_API_KEY:?WANDB_API_KEY not set}"
export WANDB_DISABLE_NETRC=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE="online"
export WANDB_DIR="$PWD/wandb_${SLURM_JOB_ID}"

mkdir -p "$WANDB_DIR"

# 按你要求打印前4后6位，确认长度
_key_head="${WANDB_API_KEY:0:4}"
_key_tail="${WANDB_API_KEY: -6}"
echo "[W&B] Using WANDB_API_KEY=${_key_head}******************************${_key_tail} (len=${#WANDB_API_KEY})"
echo "[W&B] WANDB_DIR=${WANDB_DIR}"
echo "[W&B] BASE_URL=${WANDB_BASE_URL}"

# ==== 各 rank 继承同样环境后直接启动 ====
HYDRA_FULL_ERROR=1 \
srun --export=ALL bash -lc '
  set -euo pipefail
  source /vision/u/yinhang/miniconda3/bin/activate behavior
  export KIT_DISABLE_PIP_PREBUNDLE=1
  export OMNI_KIT_DISABLE_PIP_PREBUNDLE=1
  export PYOPENGL_PLATFORM=egl
  export EGL_PLATFORM=surfaceless
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export TORCH_NCCL_BLOCKING_WAIT=1
  export NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_DEBUG=INFO
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

  echo "[Node $(hostname)] launching train.py ..."
  exec python train.py \
    data_dir=/vision/group/behavior \
    robot=r1pro task=behavior task.name=make_microwave_popcorn \
    arch=wbvima +eval=behavior \
    headless=true \
    gpus=$SLURM_NTASKS_PER_NODE num_nodes=$SLURM_NNODES bs=32 \
    trainer.num_sanity_val_steps=0 \
    '"$@"'
'

echo "Job finished."
