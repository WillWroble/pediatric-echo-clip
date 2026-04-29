#!/bin/bash
#SBATCH --job-name=fyler_posthoc
##SBATCH --partition=bch-compute-pe
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/fyler_posthoc_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/fyler_posthoc_%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2

python -u mean_pool_fyler.py \
    --scores_npz results/v10/eval_fyler_lines/fyler_scores.npz \
    --fyler_lines /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_lines.csv \
    --output results/v10/eval_fyler_lines/fyler_aurocs_posthoc.csv
