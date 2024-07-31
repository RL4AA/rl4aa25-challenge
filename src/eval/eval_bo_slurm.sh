#!/bin/sh
#SBATCH --partition=single
#SBATCH --job-name eval-bo
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type ALL
#SBATCH --output=slurm_logs/slurm_eval_bo_array_%A%a.out
#SBATCH --array=1-60


source ~/.bashrc
conda activate ares-transverse-tuning

# Move in Workspace to avoid I/O in cluster HOME
rsync -a ~/ares-transverse-tuning $(ws_find ares-transverse)/
cd $(ws_find ares-transverse)/ares-transverse-tuning


PER_TASK=5
START_NUM=$(((SLURM_ARRAY_TASK_ID - 1) * PER_TASK))
END_NUM=$((SLURM_ARRAY_TASK_ID * PER_TASK - 1))

echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Start Sweep
echo "Starting eval_bo_hard"
python -m src.eval.eval_bo_single $START_NUM $END_NUM --mode hard

echo "Starting eval_bo_proximal"
python -m src.eval.eval_bo_single $START_NUM $END_NUM --mode proximal

exit