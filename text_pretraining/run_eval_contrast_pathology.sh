#!/bin/bash
# run_eval_contrast_pathology.sh
# Submits one eval_contrast job per pathology manifest.

set -e
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining

RUN=results/full_contrast_v6
OUT=$RUN/eval_contrast_pathology
mkdir -p "$OUT"

for m in manifests/pathology/*.txt; do
    name=$(basename "$m" .txt)
    sbatch --job-name=eval_${name} eval_contrast.sbatch \
        --report_embeddings    $RUN/eval/embeddings.npz \
        --video_embeddings     $RUN/eval/embeddings_video.npz \
        --checkpoint           $RUN/latest.pt \
        --echofocus_checkpoint $RUN/echofocus_latest.pt \
        --manifest             "$m" \
        --output_dir           "$OUT"
done
