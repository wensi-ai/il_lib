#!/bin/bash
#SBATCH --job-name="train_behavior"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=490G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/sc/train_behavior_%j.out
#SBATCH --error=outputs/sc/train_behavior_%j.err

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "working directory=${SLURM_SUBMIT_DIR:-$PWD}"

# ---------- Conda ----------
source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate behavior_train
python -V

# ---------- Quiet W&B ----------
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export WANDB_DISABLE_NETRC=true
export WANDB_DISABLE_CODE=true

# ---------- Headless / EGL ----------
export CARB_APP_HEADLESS=1
export OMNI_KIT_HEADLESS=1
export KIT_DISABLE_WINDOWING=1
export OMNI_KIT_ARGS="--no-window --/app/window/enabled=false --/app/renderer/headlessRendering/enabled=true"
unset DISPLAY
export PYOPENGL_PLATFORM=egl
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# ---------- NCCL/DDP (single node) ----------
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1

# ---------- (Optional) pin these once ----------
python - <<'PY'
import subprocess, sys
def pip(*a): subprocess.check_call([sys.executable,"-m","pip",*a])
pins = ["fastapi==0.120.3","starlette==0.49.1","pydantic==2.12.3","pydantic-core==2.41.4"]
pip("install","--upgrade","--no-cache-dir",*pins)
from importlib.metadata import version; import pydantic
print("[VERIFY] fastapi =", version("fastapi"))
print("[VERIFY] pydantic =", pydantic.__version__)
print("[VERIFY] starlette =", version("starlette"))
PY

# ---------- Launch (NO unsupported overrides) ----------
HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node=8 train.py \
  data_dir=/vision/group/behavior \
  robot=r1pro \
  task=behavior \
  task.name=picking_up_trash \
  arch=wbvima \
  headless=true \
  trainer.accelerator=gpu \
  trainer.devices=8 \
  trainer.strategy=ddp \
  +trainer.limit_val_batches=0 \
  trainer.num_sanity_val_steps=0 \
  bs=32 \
  "$@"

echo "Job finished."
