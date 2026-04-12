#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/analyze_%j.out
#SBATCH --mem=256G
#SBATCH --time=01:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining

python -u analyze.py \
    --input ${INPUT:-embeddings.npz} \
    --output_fig ${OUTPUT_FIG:-analysis.png} \
    --n_studies ${N_STUDIES:-200}
