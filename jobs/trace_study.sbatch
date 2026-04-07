#!/bin/bash
#SBATCH --job-name=trace
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/trace_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

python -u /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/code/trace_study_id.py "$@"
