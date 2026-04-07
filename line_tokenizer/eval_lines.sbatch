#!/bin/bash
#SBATCH --job-name=eval_lines
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=bch-compute
#SBATCH --time=04:00:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/logs/eval_lines_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/logs/eval_lines_%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

python -u /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/eval_lines.py "$@"
