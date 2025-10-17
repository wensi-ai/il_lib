#!/bin/bash
#SBATCH --job-name="pcdvid_arr"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/sc/pcdvid_%A_%a.out
#SBATCH --error=outputs/sc/pcdvid_%A_%a.err

set -euo pipefail
mkdir -p outputs/sc

REPLAY_SCRIPT="/vision/u/yinhang/B1K_v3.7.1/BEHAVIOR-1K/OmniGibson/scripts/learning/replay_obs.py"
DATA_DIR="/vision/group/behavior/"
JOBS_TSV="${JOBS_TSV:-configs/pcd_jobs.tsv}"   # 可用 env 覆盖：JOBS_TSV=...
CHUNK=${CHUNK:-5}                               # 每个数组 task 处理多少行，可在 sbatch --export=... 里改

export OMNIGIBSON_DATASET_PATH="${DATA_DIR}"
[[ -f "${REPLAY_SCRIPT}" ]] || { echo "[ERR] no replay script: ${REPLAY_SCRIPT}"; exit 1; }
[[ -f "${JOBS_TSV}"     ]] || { echo "[ERR] no ${JOBS_TSV}"; exit 1; }

# 打印当前 pcd_range（来自 behavior.yaml），仅供确认
python - <<'PY'
import os, yaml
cfg = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        "/vision/u/yinhang/B1K_v3.7.1/BEHAVIOR-1K/OmniGibson/scripts/learning/replay_obs.py"
    ))),
    "omnigibson/learning/configs/task/behavior.yaml"
)
with open(cfg) as f:
    y = yaml.safe_load(f)
print("[pcd_range]", y.get("pcd_range"))
PY

# 计算本 task 负责的 TSV 行区间
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$(grep -c . "${JOBS_TSV}")
START=$(( ARRAY_ID * CHUNK ))
END=$(( START + CHUNK - 1 ))
if (( START >= TOTAL )); then
  echo "[SKIP] Task ${ARRAY_ID} no work (TOTAL=${TOTAL})"
  exit 0
fi
if (( END >= TOTAL )); then END=$((TOTAL-1)); fi

echo "[INFO] Task ${ARRAY_ID}: handling lines [${START}..${END}] of ${TOTAL} (CHUNK=${CHUNK})"

# 小工具：取第 n(0-based) 行的 两列 "task_name  demo_id"
get_line () {
  local idx="$1"
  awk -v n=$((idx+1)) 'BEGIN{FS="\t"} NF>=2 {i++; if(i==n){print $1"\t"$2; exit}}' "${JOBS_TSV}"
}

# 逐行处理
for i in $(seq ${START} ${END}); do
  LINE=$(get_line "${i}")
  if [[ -z "${LINE}" ]]; then
    echo "[WARN] empty TSV line at ${i}, skip"
    continue
  fi
  TASK_NAME=$(echo "${LINE}" | awk -F'\t' '{print $1}')
  DEMO_ID=$(echo "${LINE}" | awk -F'\t' '{print $2}')

  # 解析 task_id
  TASK_ID=$(python - <<PY
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
print(f"{TASK_NAMES_TO_INDICES['${TASK_NAME}']:04d}")
PY
)

  RGB_HEAD="${DATA_DIR}/2025-challenge-demos/videos/task-${TASK_ID}/observation.images.rgb.head/episode_$(printf "%08d" ${DEMO_ID}).mp4"
  DEPTH_HEAD="${DATA_DIR}/2025-challenge-demos/videos/task-${TASK_ID}/observation.images.depth.head/episode_$(printf "%08d" ${DEMO_ID}).mp4"

  if [[ ! -f "${RGB_HEAD}" || ! -f "${DEPTH_HEAD}" ]]; then
    echo "[MISS] videos not found for ${TASK_NAME} demo ${DEMO_ID}"
    echo "  ${RGB_HEAD}"
    echo "  ${DEPTH_HEAD}"
    echo "-> run once to create videos:"
    echo "   python ${REPLAY_SCRIPT} --data_folder ${DATA_DIR} --task_name ${TASK_NAME} --demo_id ${DEMO_ID} --rgbd --offline_rgbd"
    continue
  fi

  echo "[RUN] ${TASK_NAME} demo=${DEMO_ID} (task-${TASK_ID}) via --pcd_vid"
  python "${REPLAY_SCRIPT}" \
    --data_folder "${DATA_DIR}" \
    --task_name "${TASK_NAME}" \
    --demo_id "${DEMO_ID}" \
    --pcd_vid

  echo "[DONE] pcd_vid => ${DATA_DIR}/pcd_vid/task-${TASK_ID}/episode_$(printf "%08d" ${DEMO_ID}).hdf5"
done

echo "[TASK ${ARRAY_ID}] finished lines [${START}..${END}]"
