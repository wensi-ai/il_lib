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
echo "SLURM_JOB_NODELIST=${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-}"
echo "working directory=${SLURM_SUBMIT_DIR:-$PWD}"

# ========= 0) 基础环境 =========
source /vision/u/yinhang/miniconda3/bin/activate behavior

# 关闭 Isaac 的预打包 pip，使用我们 env 里的 fastapi/pydantic
export KIT_DISABLE_PIP_PREBUNDLE=1
export OMNI_KIT_DISABLE_PIP_PREBUNDLE=1

# 渲染/驱动
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# DDP / NCCL
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Hydra/FastAPI/Pydantic 版本统一（幂等）
python - <<'PY'
import sys, subprocess
def pipi(*args): subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","--no-cache-dir",*args])
pipi("fastapi>=0.110,<1.0","pydantic>=2.4,<3","starlette>=0.36,<1.0")
import fastapi, pydantic, starlette
print("[VERIFY] fastapi:", fastapi.__version__, fastapi.__file__)
print("[VERIFY] pydantic:", pydantic.__version__, pydantic.__file__)
print("[VERIFY] starlette:", starlette.__version__, starlette.__file__)
PY

# ========= 1) W&B 统一配置（提交端先定死) =========
# 必须：在提交 sbatch 时提供你的真实 key；或在这里 export 也行
# export WANDB_API_KEY="你的真实key"
: "${WANDB_API_KEY:?WANDB_API_KEY not set}"

# 强制使用可写目录（在作业提交目录下）
export WANDB_DIR="${SLURM_SUBMIT_DIR:-$PWD}/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"

# 可选：指定 Entity / Project / Run 分组
export WANDB_ENTITY="${WANDB_ENTITY:-tonyliu12345}"
export WANDB_PROJECT="${WANDB_PROJECT:-behavior-il}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-wbvima-microwave}"

# ========= 2) DDP rendezvous 显式指定 =========
# 取首个节点当 master
MASTER_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_ADDR="$MASTER_NODE"
# 避免冲突：用作业号派生端口
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

# ========= 3) 在每个任务节点启动 =========
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


  if [ "${RANK:-0}" != "0" ]; then
    export WANDB_DISABLED=true
  else
    unset WANDB_DISABLED
    python - <<PY
import os, sys, json, wandb, tempfile
key = os.environ.get("WANDB_API_KEY")
if not key:
    print("[W&B] No API key in env", file=sys.stderr); sys.exit(2)
wandb.login(key=key, relogin=True)
d = os.environ.get("WANDB_DIR")
print(f"[W&B] DIR={d}")
testfile = os.path.join(d, "._touch")
open(testfile, "w").write("ok")
os.remove(testfile)
print("[W&B] login & dir OK on rank0.")
PY
  fi

  echo "[Node $(hostname)] RANK=${RANK:-?} launching train.py ..."
  exec python train.py \
    data_dir=/vision/group/behavior \
    robot=r1pro task=behavior task.name=make_microwave_popcorn \
    arch=wbvima +eval=behavior \
    headless=true \
    gpus=$SLURM_NTASKS_PER_NODE num_nodes=$SLURM_NNODES bs=32 \
    trainer.num_sanity_val_steps=0 \
    "$@"
'

echo "Job finished."
exit 0
