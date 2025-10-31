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
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=wsai@stanford.edu

set -euo pipefail

echo "SLURM_JOBID=${SLURM_JOBID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-}"
echo "working directory=${SLURM_SUBMIT_DIR:-$PWD}"

# 0) 环境
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

# === W&B 关键：回显(掩码) 与 whoami 校验 ===
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[W&B][FATAL] WANDB_API_KEY 为空，退出"; exit 1
fi
_masked="${WANDB_API_KEY:0:6}...${WANDB_API_KEY: -4}"
echo "[W&B] Using WANDB_API_KEY=${_masked} (masked)"

# 让 wandb 忽略 ~/.netrc，避免“写 netrc / 掩码显示”干扰
export HOME="$PWD/.home_${SLURM_JOB_ID}"
mkdir -p "$HOME"

# 仅 rank0 上线、其它 offline，避免多进程同时 init
SRANK=${SLURM_PROCID:-0}
if [[ "$SRANK" == "0" ]]; then
  export WANDB_MODE=online
  export WANDB_DIR="$PWD/wandb_${SLURM_JOB_ID}"
  mkdir -p "$WANDB_DIR"
  # 显示“logged in as …”
  echo "[W&B] Checking identity via 'wandb whoami' ..."
  (WANDB_API_KEY="$WANDB_API_KEY" wandb whoami || true)
else
  export WANDB_MODE=offline
fi

# （可选：修一次 FastAPI/Pydantic 版本；如已稳定可删）
python - <<'PY'
import sys, subprocess
def pipi(*a): subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","--no-cache-dir",*a])
pipi("fastapi>=0.110,<1.0","pydantic>=2.4,<3","starlette>=0.36,<1.0")
PY

# 启动训练
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

  # 再次(掩码)回显，确保每个 rank 的日志都能看到自身环境
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[W&B][FATAL] child rank: WANDB_API_KEY 为空"; exit 1
  fi
  _masked="${WANDB_API_KEY:0:6}...${WANDB_API_KEY: -4}"
  echo "[W&B][child] Using WANDB_API_KEY=${_masked} (masked); MODE=${WANDB_MODE:-unset}; DIR=${WANDB_DIR:-unset}"

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
