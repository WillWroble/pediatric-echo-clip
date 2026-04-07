#!/bin/bash
#SBATCH --job-name=text_decoder
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_decoder/logs/text_decoder_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_decoder/logs/text_decoder_%j.err
#SBATCH --mem=128G
#SBATCH --partition=bch-gpu-pe
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_decoder

H5_PATH=${H5_PATH:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Report_Tokens/study_findings.h5}

parse_arg() {
    local key="$1"; shift
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "$key" ]]; then echo "$2"; return; fi
        shift
    done
}

OUTPUT_DIR=$(parse_arg --output_dir "$@")
EMBEDDINGS=$(parse_arg --embeddings "$@")
GENERATED="${OUTPUT_DIR}/generated_findings.json"
EVAL_DIR="${OUTPUT_DIR}/eval"

python -u train.py "$@" --h5_path "$H5_PATH"

python -u generate.py \
    --checkpoint "${OUTPUT_DIR}/latest.pt" \
    --embeddings "$EMBEDDINGS" \
    --output_dir "$OUTPUT_DIR"

python -u eval.py \
    --generated "$GENERATED" \
    --h5_path "$H5_PATH" \
    --output_dir "$EVAL_DIR"
