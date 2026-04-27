#!/bin/bash
#SBATCH --job-name=eval_fyler_probe
#SBATCH --partition=bch-compute
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/eval_fyler_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/eval_fyler_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining

python -u eval_fyler.py \
    --embeddings results/full_contrast_v6/eval/embeddings_video.npz \
    --fyler_labels /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_labels.csv \
    --fyler_lines /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_lines.csv \
    --output_dir results/full_contrast_v6/eval_fyler
