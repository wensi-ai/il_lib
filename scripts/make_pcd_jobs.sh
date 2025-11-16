#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/vision/group/behavior/"
TASK_NAME="clean_a_trumpet"          # 想跑哪个 task 就改这里（turning_on_radio = task-0000）
OUT_TSV="configs/pcd_jobs.tsv"
mkdir -p "$(dirname "$OUT_TSV")"

# 解析 4 位 task_id（不依赖外部变量展开）
TASK_ID=$(python - <<'PY'
import os
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
name = os.environ.get("TASK_NAME", "clean_a_trumpet")
print(f"{TASK_NAMES_TO_INDICES[name]:04d}")
PY
)

# 扫描 raw episodes -> (task_name \t demo_id)
ls ${DATA_DIR}/2025-challenge-rawdata/task-${TASK_ID}/episode_*.hdf5 \
| sed -E 's/.*episode_0*([0-9]+)\.hdf5/\1/' \
| awk -v task="${TASK_NAME}" '{print task"\t"$0}' > "${OUT_TSV}"

echo "[OK] wrote $(wc -l < "${OUT_TSV}") lines to ${OUT_TSV}"
head "${OUT_TSV}"
