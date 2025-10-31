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

# --- 启动训练（保持你的原参数）---
HYDRA_FULL_ERROR=1 srun python train.py \
  data_dir=/vision/group/behavior \
  robot=r1pro \
  task=behavior task.name=make_microwave_popcorn \
  arch=wbvima +eval=behavior \
  headless=true \
  gpus=$SLURM_NTASKS_PER_NODE \
  num_nodes=$SLURM_NNODES \
  bs=32 \
  trainer.num_sanity_val_steps=0 \
  "$@"

echo "Job finished."
exit 0
