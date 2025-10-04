#!/bin/bash
#SBATCH --job-name="gen_pcd_fixed"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/sc/gen_pcd_fixed_%j.out
#SBATCH --error=outputs/sc/gen_pcd_fixed_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=wsai@stanford.edu

# ---------- fixed paths ----------
REPLAY_SCRIPT="/vision/u/yinhang/B1K_v3.7.1/BEHAVIOR-1K/OmniGibson/scripts/learning/replay_obs.py"
DATA_DIR="/vision/group/behavior/"

# ---------- choose the episode here ----------
TASK_NAME="turning_on_radio"   # <-- change me: task-0000
DEMO_ID=10                    # <-- change me: from episode_00000010.hdf5
OFFLINE_RGBD=false             # true to also write rgb/depth videos
DATA_URL=""                    # optional remote bucket URL; leave blank if local raw h5 exists

# helpful job info in logs
echo "SLURM_JOBID=${SLURM_JOBID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-1}"
echo "WORKDIR=${SLURM_SUBMIT_DIR}"
nvidia-smi || true
python -V

# let OG/BEHAVIOR find metadata under DATA_DIR
export OMNIGIBSON_DATASET_PATH="${DATA_DIR}"

[[ -f "${REPLAY_SCRIPT}" ]] || { echo "[ERR] replay_obs.py not found: ${REPLAY_SCRIPT}"; exit 1; }

# replay -> rgb/depth, then fuse PCD for WB-VIMA
python "${REPLAY_SCRIPT}" \
  --data_folder "${DATA_DIR}" \
  --task_name "${TASK_NAME}" \
  --demo_id "${DEMO_ID}" \
  --rgbd $( [[ "${OFFLINE_RGBD}" == "true" ]] && echo --offline_rgbd ) \
  --pcd_gt ${DATA_URL:+--data_url "${DATA_URL}"} \
  "$@"


echo "[DONE] PCD => ${DATA_DIR}/pcd_gt/task-XXXX/episode_$(printf "%08d" ${DEMO_ID}).hdf5"
