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

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_NTASKS_PER_NODE"=$SLURM_NTASKS_PER_NODE
echo "working directory="$SLURM_SUBMIT_DIR

# 1) 激活 conda env
source /vision/u/yinhang/miniconda3/bin/activate behavior

# 2) 关闭 Isaac 的 pip 预打包，确保用到你环境里的 fastapi/pydantic
export KIT_DISABLE_PIP_PREBUNDLE=1
export OMNI_KIT_DISABLE_PIP_PREBUNDLE=1

# 3) 设定 headless 渲染（GLFW 警告就当噪音，不阻塞）
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# 4) DDP/NCCL 出错时尽快报错（别卡 30min）
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

# 5) 统一 FastAPI / Pydantic 到互相兼容的版本（路线A：Pydantic v2）
python - <<'PY'
import sys, subprocess
def pipi(*args): subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","--no-cache-dir",*args])
pipi("fastapi>=0.110,<1.0", "pydantic>=2.4,<3", "starlette>=0.36,<1.0")
print("Pinned:", __import__("fastapi").__version__, __import__("pydantic").__version__)
PY

# 6) 运行。关键：关闭 sanity-check 避免在 sanity 阶段就起 evaluator（会触发 Isaac 启动）
HYDRA_FULL_ERROR=1 \
srun python train.py \
  data_dir=/vision/group/behavior \
  robot=r1pro task=behavior task.name=make_microwave_popcorn \
  arch=wbvima +eval=behavior \
  headless=true \
  gpus=$SLURM_NTASKS_PER_NODE num_nodes=$SLURM_NNODES bs=32 \
  trainer.num_sanity_val_steps=0 \
  "$@"

echo "Job finished."
exit 0
