#!/bin/bash
#SBATCH --job-name=eval_v2
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/eval_v2_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/eval_v2_%j.err
#SBATCH --mem=128G
#SBATCH --time=02:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining

LABELS=${LABELS:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Labels_SG_Fyler_112025.csv}
DEATH_MRN=${DEATH_MRN:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/death_mrn.csv}

parse_arg() {
    local key="$1"; shift
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "$key" ]]; then echo "$2"; return; fi
        shift
    done
}

EMBEDDINGS=$(parse_arg --embeddings "$@")
H5_DIR=$(parse_arg --h5_dir "$@")
OUTPUT_DIR=$(parse_arg --output_dir "$@")

python -u eval_v2.py \
    --embeddings "$EMBEDDINGS" \
    --h5_dir "$H5_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --labels "$LABELS" \
    --death_mrn "$DEATH_MRN"

