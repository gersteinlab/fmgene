#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --out="mlp-%j.out"
#SBATCH --time=1-24:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=60G
#SBATCH --partition=pi_gerstein_gpu
#SBATCH --gpus=1

mem_bytes=$(</sys/fs/cgroup/memory/slurm/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}/memory.limit_in_bytes)
mem_gbytes=$(( $mem_bytes / 1024 **3 ))

echo "Starting at $(date)"
echo "Job submitted to the ${SLURM_JOB_PARTITION} partition on ${SLURM_CLUSTER_NAME}"
echo "${SLURM_JOB_NAME}: ${SLURM_JOB_ID}. ${SLURM_GPUS_ON_NODE} GPUs and ${mem_gbytes}GiB of RAM on compute node $(hostname)"

source /home/xz584/anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/gibbs/pi/gerstein/xz584/py3
echo "Env OK"
python main.py