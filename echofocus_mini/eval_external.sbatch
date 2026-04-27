#!/bin/bash
#SBATCH --job-name=eval_external
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/eval_external_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/eval_external_%j.err
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini

python -u eval_external.py \
    --checkpoint results/efm_jepa_meanpool_all_frac1.0/best.pt \
    --embeddings ../video_pretraining_v2/embeddings/jepa_outside.npz \
    --labels /lab-share/Cardio-Mayourian-e2/Public/Echo_Labels/echo_measurements_090425.csv \
    --label_col EF05 \
    --sid_col eid \
    --scale 100
