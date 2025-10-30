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

############################
# 0) 先在提交节点做一次环境准备（避免并发 pip）
############################
# 激活 conda env
source /vision/u/yinhang/miniconda3/bin/activate behavior

# 禁用 Isaac 的 pip 预打包，确保使用你 env 里的 fastapi/pydantic
export KIT_DISABLE_PIP_PREBUNDLE=1
export OMNI_KIT_DISABLE_PIP_PREBUNDLE=1

# Headless 渲染
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# NCCL 稳定性
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# 统一 FastAPI / Pydantic / Starlette 到兼容版本（Pydantic v2 路线）
python - <<'PY'
import sys, subprocess
def pipi(*args): subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","--no-cache-dir",*args])
pipi("fastapi>=0.110,<1.0", "pydantic>=2.4,<3", "starlette>=0.36,<1.0")
import fastapi, pydantic, starlette
print("[VERIFY] fastapi:", fastapi.__version__, fastapi.__file__)
print("[VERIFY] pydantic:", pydantic.__version__, pydantic.__file__)
print("[VERIFY] starlette:", starlette.__version__, starlette.__file__)
PY

############################
# 1) 用 srun 在每个任务上启动训练
############################
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
exit 0
