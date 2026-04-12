#!/bin/bash
#SBATCH --job-name=line_cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --partition=bch-compute-pe
#SBATCH --time=04:00:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/logs/cluster_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/logs/cluster_%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

python -u /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/cluster.py "$@"
