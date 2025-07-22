#!/bin/bash
#SBATCH --job-name="train_policy"
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --exclude=viscam1
#SBATCH --nodes=2
#SBATCH --gres=gpu:a5000:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/sc/train_policy_%j.out
#SBATCH --error=outputs/sc/train_policy_%j.err
# notifications for job done & fail
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=wsai@stanford.edu

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS"=$SLURM_GPUS
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory="$SLURM_SUBMIT_DIR

source /vision/u/wsai/miniconda3/bin/activate behavior

python train.py policy=diffusion_rgb_transformer task=turning_on_radio data_dir=/vision/u/wsai/behavior gpus=$SLURM_GPUS num_nodes=$SLURM_NNODES

echo "Job finished."
exit 0
