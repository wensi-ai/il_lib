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

# ========== W&B 变量（不写你的 ~/.netrc）==========
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE=online
export WANDB_NO_NETRC=1
export WANDB_NO_NETRC_WRITES=1
export WANDB_NETRC_PATH=/dev/null
export WANDB_CONFIG_DIR="$PWD/.wandb_cfg_${SLURM_JOB_ID}"
mkdir -p "$WANDB_CONFIG_DIR"

# 可选：固定 entity / project（不需要可注释）
# export WANDB_ENTITY="tonyliu12345"
# export WANDB_PROJECT="behavior-il"

# 必须有 API KEY
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[W&B] ERROR: WANDB_API_KEY 为空（请在提交端 export 再 sbatch）"; exit 1
fi

# 打印 key 前4后6位
_key_fmt="$(python - <<'PY'
import os
k=os.environ.get("WANDB_API_KEY","").strip()
print(k[:4] + ("*"*max(len(k)-10,0)) + k[-6:])
PY
)"
echo "[W&B] Using WANDB_API_KEY=${_key_fmt}"
echo "[W&B] BASE_URL=${WANDB_BASE_URL}"

export WANDB_DIR="$PWD/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
echo "[W&B] WANDB_DIR=${WANDB_DIR}"

# 依赖版本
python -m pip install --upgrade --no-cache-dir wandb requests >/dev/null
python - <<'PY'
import sys,subprocess
def pipi(*a): subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","--no-cache-dir",*a])
pipi("fastapi>=0.110,<1.0","pydantic>=2.4,<3","starlette>=0.36,<1.0")
import fastapi,pydantic,starlette
print(f"[VERIFY] fastapi={fastapi.__version__}  pydantic={pydantic.__version__}  starlette={starlette.__version__}")
PY

# ========== 每个 rank：网络连通 + 原生 HTTP 验证 token ==========
HYDRA_FULL_ERROR=1 \
srun --export=ALL bash -lc '
  set -euo pipefail
  source /vision/u/yinhang/miniconda3/bin/activate behavior
  python -m pip install --upgrade --no-cache-dir wandb requests >/dev/null

  _host="${WANDB_BASE_URL:-https://api.wandb.ai}"
  _key="${WANDB_API_KEY:-}"

  # 打印 key 片段
  _key_fmt=$(python - <<'"PY"'
import os
k=os.environ.get("WANDB_API_KEY","").strip()
print(k[:4] + ("*"*max(len(k)-10,0)) + k[-6:])
PY
)
  echo "[W&B][child $(hostname)] KEY=${_key_fmt} HOST=${_host} DIR=${WANDB_DIR:-}"

  # 1) 基础连通性：DNS + TLS + HTTP
  python - <<PY
import os, sys, socket, ssl, requests
host=os.environ.get("WANDB_BASE_URL","https://api.wandb.ai")
host_name=host.split("://",1)[1].split("/",1)[0]
try:
    ip=socket.gethostbyname(host_name)
    print(f"[NET] DNS {host_name} -> {ip}")
except Exception as e:
    print("[NET] DNS failed:", e); sys.exit(10)
try:
    ctx=ssl.create_default_context(); s=socket.create_connection((host_name,443),timeout=5)
    with ctx.wrap_socket(s, server_hostname=host_name) as ss:
        print("[NET] TLS ok:", ss.version())
except Exception as e:
    print("[NET] TLS failed:", e); sys.exit(11)
try:
    r=requests.get(host+"/status", timeout=10)
    print("[NET] GET /status ->", r.status_code, r.text[:80].replace("\n"," "))
except Exception as e:
    print("[NET] HTTP failed:", e); sys.exit(12)
PY

  # 2) 原生 GraphQL 验证 token
  python - <<PY
import os, sys, requests, json
host=os.environ.get("WANDB_BASE_URL","https://api.wandb.ai").rstrip("/")
key=os.environ.get("WANDB_API_KEY","").strip()
q={"query":"{ viewer { entity username email } }"}
try:
    resp=requests.post(host+"/graphql", json=q, timeout=15,
                       headers={"Authorization": f"Bearer {key}"})
    print("[W&B][HTTP] POST /graphql ->", resp.status_code)
    if resp.status_code!=200:
        print("[W&B][HTTP] body:", resp.text[:200])
        sys.exit(20)
    data=resp.json()
    if "errors" in data:
        print("[W&B][HTTP] GraphQL errors:", data["errors"])
        sys.exit(21)
    v=data["data"]["viewer"]
    print(f"[W&B][HTTP] Viewer ok: entity={v.get("entity") or v.get("username")} email={v.get("email")}")
except Exception as e:
    print("[W&B][HTTP] request failed:", e); sys.exit(22)
PY

  # 3) SDK 登录（指定 host，避免走默认/写 netrc）
  python - <<PY
import os, sys, wandb
key=os.environ.get("WANDB_API_KEY","").strip()
host=os.environ.get("WANDB_BASE_URL","https://api.wandb.ai").strip()
ok=wandb.login(key=key, relogin=True, host=host)
print("[W&B][SDK] wandb.login ->", ok)
if not ok: sys.exit(30)
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
