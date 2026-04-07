#!/bin/bash
#SBATCH --job-name=train_embed_eval
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/train_embed_eval_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/train_embed_eval_%j.err
#SBATCH --mem=256G
#SBATCH --partition=bch-gpu-pe
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining

LABELS=${LABELS:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Labels_SG_Fyler_112025.csv}
DEATH_MRN=${DEATH_MRN:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/death_mrn.csv}
LVEF_LABELS=${LVEF_LABELS:-/lab-share/Cardio-Mayourian-e2/Public/EchoFocus_Features/Bill_Features/Subgroup_Details_from_ECGLVEF.csv}
parse_arg() {
    local key="$1"; shift
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "$key" ]]; then echo "$2"; return; fi
        shift
    done
}

H5_DIR=$(parse_arg --h5_dir "$@")
OUTPUT_DIR=$(parse_arg --output_dir "$@")
EMBEDDINGS="${OUTPUT_DIR}/eval/embeddings.npz"
EVAL_DIR="${OUTPUT_DIR}/eval"
EVAL_V2_DIR="${OUTPUT_DIR}/eval_v2"
EVAL_SUPERVISED_DIR="${OUTPUT_DIR}/eval_supervised"
CHECKPOINT="${OUTPUT_DIR}/latest.pt"
VIDEO_EMBEDDINGS=$(parse_arg --video_embeddings "$@")
STUDY_EMBEDDINGS=$(parse_arg --study_embeddings "$@")
EMBED_EXTRA=""
if [[ -n "$VIDEO_EMBEDDINGS" ]]; then
    EMBED_EXTRA="--echofocus_checkpoint ${OUTPUT_DIR}/echofocus_latest.pt --video_embeddings ${VIDEO_EMBEDDINGS}"
fi

python -u train.py "$@"

python -u embed.py \
    --checkpoint "$CHECKPOINT" \
    --h5_dir "$H5_DIR" \
    --train_manifest "$(parse_arg --train_manifest "$@")" \
    --val_manifest "$(parse_arg --val_manifest "$@")" \
    --output "$EMBEDDINGS" \
    $EMBED_EXTRA
if [[ -f "${OUTPUT_DIR}/eval/embeddings_video.npz" ]]; then
    STUDY_EMBEDDINGS="${OUTPUT_DIR}/eval/embeddings_video.npz"
fi


python -u eval.py \
    --report_embeddings "$EMBEDDINGS" \
    ${STUDY_EMBEDDINGS:+--study_embeddings "$STUDY_EMBEDDINGS"} \
    --h5_dir "$H5_DIR" \
    --output_dir "$EVAL_DIR" \
    --labels "$LABELS" \
    --death_mrn "$DEATH_MRN"

python -u eval_supervised.py \
    --report_embeddings "$EMBEDDINGS" \
    ${STUDY_EMBEDDINGS:+--study_embeddings "$STUDY_EMBEDDINGS"} \
    --h5_dir "$H5_DIR" \
    --output_dir "$EVAL_SUPERVISED_DIR" \
    --labels "$LABELS" \
    --lvef_labels "$LVEF_LABELS" \
    --death_mrn "$DEATH_MRN"
