"""K-NN table for study and report embeddings.

Outputs two CSVs (video_knn.csv, report_knn.csv) sorted by MRN.
Columns: MRN, study_id, neighbor_1_MRN, neighbor_1_study_id, neighbor_1_sim, ...

Usage:
    python -u knn_table.py \
        --report_npz results/full_contrast_v6/eval/embeddings.npz \
        --video_npz  results/full_contrast_v6/eval/embeddings_video.npz \
        --mrn_csv    /path/to/echo_reports_v3.csv \
        --output_dir results/full_contrast_v6/eval/knn \
        --k 10 --split val --batch_size 512
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_mrn_map(csv_path):
    """Auto-detect study_id and MRN columns from CSV."""
    df = pd.read_csv(csv_path, dtype=str, usecols=lambda c: c.lower() in (
        "study_id", "event.id.number", "eid", "mrn",
    ))
    cols_lower = {c.lower(): c for c in df.columns}

    sid_col = None
    for candidate in ["study_id", "event.id.number", "eid"]:
        if candidate in cols_lower:
            sid_col = cols_lower[candidate]
            break

    mrn_col = cols_lower.get("mrn")
    if sid_col is None or mrn_col is None:
        raise ValueError(f"Cannot find study_id / MRN columns in {csv_path}. "
                         f"Found: {list(df.columns)}")

    df = df[[sid_col, mrn_col]].drop_duplicates(subset=sid_col)
    return dict(zip(df[sid_col].astype(str), df[mrn_col].astype(str)))


def batched_topk(Z, k, batch_size, device):
    """Cosine top-K via batched dot product on L2-normed embeddings.

    Returns:
        sims: (N, K) float32 cosine similarities
        idxs: (N, K) int64 neighbor indices
    """
    N = Z.shape[0]
    Z_t = torch.from_numpy(Z).to(device)
    Z_t = F.normalize(Z_t, dim=1)

    all_sims = torch.empty(N, k, device="cpu")
    all_idxs = torch.empty(N, k, dtype=torch.long, device="cpu")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sim = Z_t[start:end] @ Z_t.T  # (B, N)

        # mask self
        for i in range(start, end):
            sim[i - start, i] = -1.0

        topk_sim, topk_idx = sim.topk(k, dim=1)
        all_sims[start:end] = topk_sim.cpu()
        all_idxs[start:end] = topk_idx.cpu()

        if start % (batch_size * 20) == 0:
            print(f"  {start:,}/{N:,}", flush=True)

    return all_sims.numpy(), all_idxs.numpy()


def build_knn_csv(study_ids, sims, idxs, mrn_map, k):
    """Build DataFrame: study_id | MRN | nn_1 .. nn_K."""
    rows = []
    for i, sid in enumerate(study_ids):
        row = {"study_id": sid, "MRN": mrn_map.get(sid, "")}
        for j in range(k):
            row[f"nn_{j+1}"] = study_ids[idxs[i, j]]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report_npz", required=True)
    p.add_argument("--video_npz", required=True)
    p.add_argument("--mrn_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # MRN lookup
    print("Loading MRN map...", flush=True)
    mrn_map = load_mrn_map(args.mrn_csv)
    print(f"  {len(mrn_map):,} study→MRN entries", flush=True)

    for tag, npz_path in [("report", args.report_npz), ("video", args.video_npz)]:
        print(f"\n=== {tag} embeddings ===", flush=True)
        data = np.load(npz_path)
        Z = data[args.split].astype(np.float32)
        study_ids = data[f"{args.split}_ids"].astype(str).tolist()
        print(f"  {len(study_ids):,} studies, {Z.shape[1]}d", flush=True)

        n_mapped = sum(1 for s in study_ids if s in mrn_map)
        print(f"  MRN coverage: {n_mapped:,}/{len(study_ids):,}", flush=True)

        print("  Computing top-K...", flush=True)
        sims, idxs = batched_topk(Z, args.k, args.batch_size, device)

        df = build_knn_csv(study_ids, sims, idxs, mrn_map, args.k)
        csv_path = out / f"{tag}_knn.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path} ({len(df):,} rows)", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
