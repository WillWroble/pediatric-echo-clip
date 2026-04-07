#!/bin/bash
#SBATCH --job-name=umap_echo
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/umap_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/umap_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
python /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/code/umap_embeddings.py
