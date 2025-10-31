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

# ==== 0) 环境 ====
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

# 统一 fastapi / pydantic / starlette 版本（Pydantic v2 路线）
python - <<'PY'
import sys, subprocess
def pipi(*args): subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","--no-cache-dir",*args])
pipi("fastapi>=0.110,<1.0", "pydantic>=2.4,<3", "starlette>=0.36,<1.0")
import fastapi, pydantic, starlette
print("[VERIFY] fastapi=", fastapi.__version__, " pydantic=", pydantic.__version__, " starlette=", starlette.__version__)
PY

# ==== 1) W&B：父进程打印关键环境并校验 ====
# 你可以按需设置，非必需：
# export WANDB_ENTITY="tonyliu12345"
# export WANDB_PROJECT="behavior-il"

# 让每个作业有独立的 W&B 目录
export WANDB_DIR="$PWD/wandb_${SLURM_JOBID:-$$}"
mkdir -p "$WANDB_DIR"

# 父进程打印 API KEY（前4后6）
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[W&B] Using WANDB_API_KEY=${WANDB_API_KEY:0:4}******************************${WANDB_API_KEY: -6}"
else
  echo "[W&B] WANDB_API_KEY is EMPTY!"; exit 3
fi
echo "[W&B] WANDB_DIR=${WANDB_DIR}"
echo "[W&B] BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}"

# 父进程做一次网络与身份自检（GraphQL viewer）
python - <<'PY'
import os, socket, ssl, json, urllib.request, urllib.error
base = os.environ.get("WANDB_BASE_URL","https://api.wandb.ai").rstrip("/")
key  = os.environ.get("WANDB_API_KEY","")
def mask(s): return f"{s[:4]}{'*'*30}{s[-6:]}" if len(s)>=10 else s
print(f"[W&B][PARENT] KEY={mask(key)}")
host = base.replace("https://","").split("/")[0]
ip = socket.gethostbyname(host)
print("[NET] DNS", host, "->", ip)
ctx = ssl.create_default_context()
with socket.create_connection((host, 443), timeout=5) as sock:
    with ctx.wrap_socket(sock, server_hostname=host) as ssock:
        print("[NET] TLS ok:", ssock.version())
try:
    with urllib.request.urlopen(base + "/status", timeout=5) as r:
        body = r.read(100).decode("utf-8","ignore").strip()
        print("[NET] GET /status ->", r.status, body)
except urllib.error.HTTPError as e:
    print("[NET] GET /status ->", e.code, (e.read(100) or b"").decode("utf-8","ignore").strip())
req = urllib.request.Request(base + "/graphql",
    data=json.dumps({"query":"query { viewer { entity username email } }"}).encode("utf-8"),
    headers={"Content-Type":"application/json","Authorization":f"Bearer {key}"},
    method="POST")
try:
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8"))
        if "errors" in data:
            print("[W&B][PARENT] viewer() errors:", data["errors"])
        else:
            v = data["data"]["viewer"]
            print(f"[W&B][PARENT] Viewer ok: entity={v.get('entity') or v.get('username')} email={v.get('email')}")
except urllib.error.HTTPError as e:
    print("[W&B][PARENT] viewer() HTTPError:", e.code, (e.read(200) or b"").decode("utf-8","ignore").strip())
except Exception as e:
    print("[W&B][PARENT] viewer() FAILED:", type(e).__name__, str(e))
PY

# ==== 2) srun 启动每个 rank，子进程也打印 Key 片段并自检 ====
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

  # 每个 rank 的 W&B 目录
  export WANDB_DIR="${WANDB_DIR}/rank_${SLURM_PROCID:-0}"
  mkdir -p "$WANDB_DIR"

  # 子进程打印 API Key（前4后6）
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[W&B][child $(hostname)] KEY=${WANDB_API_KEY:0:4}******************************${WANDB_API_KEY: -6} HOST=${WANDB_BASE_URL:-https://api.wandb.ai} DIR=$WANDB_DIR"
  else
    echo "[W&B][child] WANDB_API_KEY EMPTY!"; exit 3
  fi

  # 子进程轻量身份校验（GraphQL viewer）
  python - <<'"'"'PY'"'"'
import os, json, urllib.request
base = os.environ.get("WANDB_BASE_URL","https://api.wandb.ai").rstrip("/")
key  = os.environ.get("WANDB_API_KEY","")
def mask(s): return f"{s[:4]}{'*'*30}{s[-6:]}" if len(s)>=10 else s
print(f"[W&B][child] KEY={mask(key)} BASE={base}")
req = urllib.request.Request(base + "/graphql",
    data=json.dumps({"query":"query { viewer { entity username email } }"}).encode("utf-8"),
    headers={"Content-Type":"application/json","Authorization":f"Bearer {key}"},
    method="POST")
try:
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8"))
        if "errors" in data:
            print("[W&B][child] viewer() errors:", data["errors"])
        else:
            v = data["data"]["viewer"]
            print(f"[W&B][child] Viewer ok: entity={v.get('entity') or v.get('username')} email={v.get('email')}")
except Exception as e:
    print("[W&B][child] viewer() FAILED:", type(e).__name__, str(e))
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
exit 0
