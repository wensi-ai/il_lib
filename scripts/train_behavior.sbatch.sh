#!/bin/bash
#SBATCH --job-name="train_behavior"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=490G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/sc/train_behavior_%j.out
#SBATCH --error=outputs/sc/train_behavior_%j.err

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "working directory=${SLURM_SUBMIT_DIR:-$PWD}"

# ==== Conda ====
source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate behavior_train
python -V
python - <<'PY'
import sys, importlib.metadata as m
import fastapi, pydantic
print("PYTHON =", sys.executable)
print("fastapi =", fastapi.__version__, " pydantic =", pydantic.__version__, " starlette =", m.version("starlette"))
PY

# ==== 禁用 W&B ====
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export WANDB_DISABLE_NETRC=true
export WANDB_DISABLE_CODE=true

# ==== 分布式 / NCCL / 网卡 ====
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eno1     # 你的内网网卡（已验证）
export OMP_NUM_THREADS=8
export PL_TORCH_DISTRIBUTED_TIMEOUT=600

# ==== 彻底无头（禁止 windowing） ====
export CARB_APP_HEADLESS=1
export OMNI_KIT_HEADLESS=1
export KIT_DISABLE_WINDOWING=1
export OMNI_KIT_ARGS="--no-window --/app/window/enabled=false --/app/renderer/headlessRendering/enabled=true"
unset DISPLAY
export PYOPENGL_PLATFORM=egl
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# ==== 精准 pin FastAPI / Pydantic / Starlette（一次性、幂等） ====
python - <<'PY'
import subprocess, sys
def pip(*args): subprocess.check_call([sys.executable, "-m", "pip"]+list(args))
pins = [
  "fastapi==0.120.3",
  "starlette==0.49.1",
  "pydantic==2.12.3",
  "pydantic-core==2.41.4",
]
pip("install","--upgrade","--no-cache-dir", *pins)
from importlib.metadata import version
import pydantic
print("[VERIFY] fastapi =", version("fastapi"))
print("[VERIFY] pydantic =", pydantic.__version__)
print("[VERIFY] starlette =", version("starlette"))
PY

# ==== 启动训练（禁用 evaluator/验证） ====
HYDRA_FULL_ERROR=1 srun python train.py \
  data_dir=/vision/group/behavior \
  robot=r1pro \
  task=behavior \
  task.name=picking_up_trash \
  arch=wbvima \
  headless=true \
  gpus=$SLURM_NTASKS_PER_NODE \
  num_nodes=$SLURM_NNODES \
  bs=32 \
  +trainer.limit_val_batches=0 \
  trainer.num_sanity_val_steps=0 \
  "$@"


echo "Job finished."
