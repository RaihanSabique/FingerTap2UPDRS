#!/bin/bash
#SBATCH --job-name=ht-pose
#SBATCH --partition=batch
#SBATCH --array=0-39
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out

source /etc/profile.d/modules.sh
module load python/3.10.13     # site-specific: change or drop on other clusters
# submit from the repo root (mkdir -p slurm/logs; sbatch slurm/<name>.sbatch)
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=4
python scripts/extract_pose.py --shard $SLURM_ARRAY_TASK_ID --n_shards 40 --threads 4
