#!/bin/bash
#SBATCH --job-name=extract_fyler
#SBATCH --partition=bch-compute
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/extract_fyler_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/extract_fyler_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip
python -u code/extract_fyler.py
