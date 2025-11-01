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
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=wsai@stanford.edu

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "working directory=${SLURM_SUBMIT_DIR:-$PWD}"

# --- Conda env ---
source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate behavior_train
python -V
python -c "import sys; print(sys.executable)"
python -c "import fastapi, pydantic; print('fastapi=', fastapi.__version__, 'pydantic=', pydantic.__version__)"

# --- 禁用 W&B（完全离线）---
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export WANDB_DISABLE_NETRC=true
export WANDB_DISABLE_CODE=true

# ===== 分布式与无头环境（单节点优化） =====
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PL_TORCH_DISTRIBUTED_TIMEOUT=600
export PYTHONFAULTHANDLER=1

# 单节点建议开启 P2P / 关闭 IB（没有 IB 也无所谓）
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# 纯无头渲染（Isaac/Omni）
export OMNI_KIT_HEADLESS=1
export PYOPENGL_PLATFORM=egl
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset DISPLAY

# （可选）CPU 绑定更稳
export OMP_NUM_THREADS=8

# --- 精准 pin FastAPI / Pydantic / Starlette 以修复不兼容 ---
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

# --- 启动训练（单节点）---
# 说明：
# - gpus 使用本节点任务数（= 卡数）
# - num_nodes 固定为 1（关键）
# - 若从 2 节点*4卡 切到 1 节点*8卡，等效全局 batch 差不多；不够就用 grad_acc 叠上去
HYDRA_FULL_ERROR=1 srun python train.py \
  data_dir=/vision/group/behavior \
  robot=r1pro \
  task=behavior task.name=make_microwave_popcorn \
  arch=wbvima +eval=behavior \
  headless=true \
  gpus=$SLURM_NTASKS_PER_NODE \
  num_nodes=1 \
  bs=32 \
  trainer.num_sanity_val_steps=0 \
  "$@"

echo "Job finished."
exit 0
