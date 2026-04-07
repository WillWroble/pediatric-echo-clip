#!/bin/bash
#SBATCH --job-name=bert_embed
#SBATCH --array=0-697
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/bert_embed_%a.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/bert_embed_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

python /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/code/embed_lines.py $SLURM_ARRAY_TASK_ID

