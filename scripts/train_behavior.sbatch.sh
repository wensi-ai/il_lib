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

# ========== 环境 ==========
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

# ========== W&B 基础 ==========
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE=online
export WANDB_NO_NETRC_WRITES=1  # 不改你的 ~/.netrc
# 可选：固定 entity / project（无需要可注释掉）
# export WANDB_ENTITY="tonyliu12345"
# export WANDB_PROJECT="behavior-il"

# 优先用环境变量，其次从 ~/.netrc 读
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(python - <<'PY'
import os,re
p=os.path.expanduser("~/.netrc")
try:
    t=open(p).read()
    m=re.search(r'machine\s+api\.wandb\.ai\s+login\s+\S+\s+password\s+(\S+)',t,re.I|re.M)
    print(m.group(1) if m else "", end="")
except FileNotFoundError:
    pass
PY
)"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[W&B] ERROR: 没有拿到 API Key（环境变量和 ~/.netrc 都空）。"
  exit 1
fi

# 明确打印前4后6位（不中间加 xxxx，那串星号只是遮位长度）
_key_fmt="$(python - <<'PY'
import os
k=os.environ.get("WANDB_API_KEY","")
print(k[:4] + ("*"*max(len(k)-10,0)) + k[-6:])
PY
)"
echo "[W&B] Using WANDB_API_KEY=${_key_fmt}"

export WANDB_DIR="$PWD/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
echo "[W&B] WANDB_DIR=${WANDB_DIR}"
echo "[W&B] BASE_URL=${WANDB_BASE_URL}"

# 主节点升级 wandb（子节点也会再升级一次，保证版本一致）
python -m pip install --upgrade --no-cache-dir wandb >/dev/null

# 依赖版本钉住
python - <<'PY'
import sys,subprocess
def pipi(*a): subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","--no-cache-dir",*a])
pipi("fastapi>=0.110,<1.0","pydantic>=2.4,<3","starlette>=0.36,<1.0")
import fastapi,pydantic,starlette
print(f"[VERIFY] fastapi={fastapi.__version__}  pydantic={pydantic.__version__}  starlette={starlette.__version__}")
PY

# ========== 在每个 rank 本地强制登录 + 验身 ==========
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

  # 子进程也升级一次 wandb，避免版本不一致
  python -m pip install --upgrade --no-cache-dir wandb >/dev/null

  # 打印 key 片段，确保每个 rank 都拿到同一个 key
  _key_fmt=$(python - <<'"PY"'
import os
k=os.environ.get("WANDB_API_KEY","")
print(k[:4] + ("*"*max(len(k)-10,0)) + k[-6:])
PY
)
  echo "[W&B][child $(hostname)] API_KEY=${_key_fmt}; MODE=${WANDB_MODE:-}; DIR=${WANDB_DIR:-}"

  # **关键**：用 SDK 在本机内强制登录 + 获取 viewer，失败就立刻退出
  python - <<'"PY"'
import os, sys
import wandb
from wandb.apis.public import Api
key=os.environ.get("WANDB_API_KEY","").strip()
if not key:
    print("[W&B][child] ERROR: WANDB_API_KEY missing"); sys.exit(2)
ok=wandb.login(key=key, relogin=True)
print("[W&B][child] wandb.login ->", ok)
try:
    u=Api().viewer()
    # u 是 GQL 对象，取常见字段
    ent=getattr(u, "entity", None) or getattr(u, "username", None)
    mail=getattr(u, "email", None)
    print(f"[W&B][child] Logged in as entity={ent} email={mail}")
except Exception as e:
    print("[W&B][child] viewer() failed:", e); sys.exit(3)
PY

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
