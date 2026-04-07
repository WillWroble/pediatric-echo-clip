#!/bin/bash
#SBATCH --job-name=encode_study
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/encode_study_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/encode_study_%j.err
#SBATCH --mem=512G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip
parse_arg() {
    local key="$1"; shift
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "$key" ]]; then echo "$2"; return; fi
        shift
    done
}
PROBE_WEIGHTS=$(parse_arg --probe_weights "$@")
LABELS=${LABELS:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Labels_SG_Fyler_112025.csv}
DEATH_MRN=${DEATH_MRN:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/death_mrn.csv}
HDF5_DIR=$(parse_arg --hdf5_dir "$@")
VP_CKPT=$(parse_arg --vp_ckpt "$@")
EF_CKPT=$(parse_arg --ef_ckpt "$@")
OUTPUT_DIR=$(parse_arg --output_dir "$@")
MANIFEST=$(parse_arg --manifest "$@")
MANIFEST_ARG=""
if [[ -n "$MANIFEST" ]]; then
    MANIFEST_ARG="--manifest $MANIFEST"
fi
mkdir -p "$OUTPUT_DIR"
python -u video_pretraining/embed.py \
    --hdf5_dir "$HDF5_DIR" \
    --checkpoint "$VP_CKPT" \
    $MANIFEST_ARG \
    --output "${OUTPUT_DIR}/video_embeddings.npz"
python -u echofocus_mini/embed.py \
    --checkpoint "$EF_CKPT" \
    --embeddings "${OUTPUT_DIR}/video_embeddings.npz" \
    --output "${OUTPUT_DIR}/study_embeddings.npz"
python -u text_pretraining/eval.py \
    --study_embeddings "${OUTPUT_DIR}/study_embeddings.npz" \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings \
    --labels "$LABELS" \
    --death_mrn "$DEATH_MRN" \
    --output_dir "${OUTPUT_DIR}/eval"

python -u text_pretraining/eval_supervised.py \
    --study_embeddings "${OUTPUT_DIR}/study_embeddings.npz" \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings \
    --labels "$LABELS" \
    --death_mrn "$DEATH_MRN" \
    ${PROBE_WEIGHTS:+--study_probe_weights "$PROBE_WEIGHTS"} \
    --output_dir "${OUTPUT_DIR}/eval_supervised"
