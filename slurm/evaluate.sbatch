#!/bin/bash
#SBATCH --job-name=ht-eval
#SBATCH --partition=batch
#SBATCH --array=0-131
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out

source /etc/profile.d/modules.sh
module load python/3.10.13     # site-specific: change or drop on other clusters
# submit from the repo root (mkdir -p slurm/logs; sbatch slurm/<name>.sbatch)
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
python scripts/evaluate.py --job $SLURM_ARRAY_TASK_ID --n_jobs 8
