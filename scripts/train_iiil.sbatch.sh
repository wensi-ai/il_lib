#!/bin/bash
#SBATCH --job-name="train_iiil"
#SBATCH --account=viscam
#SBATCH --partition=svl,viscam
#SBATCH --exclude=svl12,svl13
#SBATCH --nodes=2
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=230G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/sc/train_iiil_%j.out
#SBATCH --error=outputs/sc/train_iiil_%j.err
# notifications for job done & fail
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=wsai@stanford.edu

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_CPU_PER_TASK="$SLURM_CPUS_PER_TASK
echo "SLURM_MEM_PER_NODE="$SLURM_MEM_PER_NODE
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_NTASKS_PER_NODE"=$SLURM_NTASKS_PER_NODE
echo "working directory="$SLURM_SUBMIT_DIR

source /vision/u/wsai/miniconda3/bin/activate behavior

OMNIGIBSAON_NO_OMNI_LOGS=1 srun python train.py +eval=iiil gpus=$SLURM_NTASKS_PER_NODE num_nodes=$SLURM_NNODES "$@"

echo "Job finished."
exit 0
