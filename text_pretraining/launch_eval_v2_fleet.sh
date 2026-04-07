#!/bin/bash
# Launch eval_v2 jobs for all results folders that have embeddings.npz.
# Usage: bash launch_eval_v2_fleet.sh
#
# Optionally override defaults:
#   RESULTS_DIR=/path/to/results H5_DIR=/path/to/h5 bash launch_eval_v2_fleet.sh

RESULTS_DIR=${RESULTS_DIR:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/results}
H5_DIR=${H5_DIR:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings}

n_submitted=0
n_skipped=0

for embeddings in "$RESULTS_DIR"/*/eval/embeddings.npz; do
    eval_dir=$(dirname "$embeddings")            # results/vicreg_v30_nofetal/eval
    output_dir="${eval_dir%/eval}/eval_v2"       # results/vicreg_v30_nofetal/eval_v2

    if [[ -f "${output_dir}/umap_3d.html" ]]; then
        echo "SKIP  $eval_dir  (already done)"
        ((n_skipped++))
        continue
    fi

    echo "SUBMIT $eval_dir"
    sbatch eval_v2.sbatch \
        --embeddings "$embeddings" \
        --h5_dir "$H5_DIR" \
        --output_dir "$output_dir"
    ((n_submitted++))
done

echo ""
echo "Submitted: $n_submitted  Skipped: $n_skipped"
