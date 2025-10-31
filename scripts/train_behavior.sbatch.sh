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

# ===== 0) 环境 =====
source /vision/u/yinhang/miniconda3/bin/activate behavior

# Isaac / OpenGL / NCCL
export KIT_DISABLE_PIP_PREBUNDLE=1
export OMNI_KIT_DISABLE_PIP_PREBUNDLE=1
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

# ===== 1) W&B 登录（优先环境变量；否则读取 ~/.netrc；不写 netrc）=====
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE=online
export WANDB_NO_NETRC_WRITES=1   # 不去改你的 ~/.netrc

# 从 env 或 netrc 拿 key
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  # 回退：从 ~/.netrc 里读取 api.wandb.ai 的 password
  WANDB_API_KEY="$(python - <<'PY'
import os, re
p=os.path.expanduser("~/.netrc")
try:
    with open(p,"r") as f:
        t=f.read()
    m=re.search(r'machine\s+api\.wandb\.ai\s+login\s+\S+\s+password\s+(\S+)', t, re.I|re.M)
    if m:
        print(m.group(1))
except FileNotFoundError:
    pass
PY
)"
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[W&B] ERROR: 找不到 API Key。请在提交前先：export WANDB_API_KEY='你的真实KEY'，或确保 ~/.netrc 已配置。"
  exit 1
fi

# 打印真实 key 的前4后6位（中间不遮罩为xxxx，而是直接样式化）
_key_fmt="$(python - <<'PY'
import os
k=os.environ.get("WANDB_API_KEY","")
print(k[:4] + (k[4:-6] and "*"*len(k[4:-6]) or "") + k[-6:])
PY
)"
echo "[W&B] Using WANDB_API_KEY=${_key_fmt}"

# 独立目录，避免权限/并发
export WANDB_DIR="$PWD/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
echo "[W&B] WANDB_DIR=${WANDB_DIR}"
echo "[W&B] BASE_URL=${WANDB_BASE_URL}"

# 升级 wandb，兼容无 'whoami' 的旧版本；用 CLI 显式登录，这样日志会出现 “Logged in as …”
python -m pip install --upgrade --no-cache-dir wandb >/dev/null
wandb login --relogin "$WANDB_API_KEY"

# (可选) 固定 entity / project；不需要可注释
# export WANDB_ENTITY="tonyliu12345"
# export WANDB_PROJECT="behavior-il"

# ===== 2) 版本兼容（FastAPI / Pydantic / Starlette）=====
python - <<'PY'
import sys, subprocess
def pipi(*args): subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","--no-cache-dir",*args])
pipi("fastapi>=0.110,<1.0","pydantic>=2.4,<3","starlette>=0.36,<1.0")
import fastapi, pydantic, starlette
print(f"[VERIFY] fastapi={fastapi.__version__}  pydantic={pydantic.__version__}  starlette={starlette.__version__}")
PY

# ===== 3) 启动训练 =====
HYDRA_FULL_ERROR=1 \
srun --export=ALL bash -lc '
  set -euo pipefail
  source /vision/u/yinhang/miniconda3/bin/activate behavior

  export KIT_DISABLE_PIP_PREBUNDLE=1
  export OMNI_KIT_DISABLE_PIP_PREBUNDLE=1
  export PYOPENGL_PLATFORM=egl
  export EGL_PLATFORM=surfaceless
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
  export TORCH_NCCL_BLOCKING_WAIT=1
  export NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_DEBUG=INFO

  # 子进程也打印一次（前4后6）
  _key_fmt=$(python - <<'"PY"'
import os
k=os.environ.get("WANDB_API_KEY","")
print(k[:4] + (k[4:-6] and "*"*len(k[4:-6]) or "") + k[-6:])
PY
)
  echo "[W&B][child] API_KEY=${_key_fmt}; MODE=${WANDB_MODE:-}; DIR=${WANDB_DIR:-}"

  echo "[Node $(hostname)] launching train.py ..."
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
