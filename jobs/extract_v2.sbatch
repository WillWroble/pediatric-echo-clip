#!/bin/bash
#SBATCH --job-name=echo_extract_v2
#SBATCH --array=0-697
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/extract_v2_%a.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/extract_v2_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

python /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/code/extract_v2.py $SLURM_ARRAY_TASK_ID

