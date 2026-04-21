"""Line embedding UMAP visualization with Fyler subgroup analysis.

Usage:
    python -u eval_lines_v2.py \
        --pool_dir results/v2/pool_cache \
        --h5_dir /lab-share/.../line_tokenizer/data \
        --labels /lab-share/.../Echo_Labels_SG_Fyler_112025.csv \
        --output_dir results/v2/eval_lines_v2
"""

import argparse
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from umap import UMAP


DIAG_COLS = [
    "Composite_Critical", "Composite_NonCritical",
    "VSD", "ASD", "TOF", "AVCD", "Coarct", "HLHS", "PDA",
    "TOF_ADJ", "AVCD_ADJ", "VSD_ADJ", "Coarct_ADJ",
]


def merge_soft_wraps(lines):
    merged = []
    for line in lines:
        if merged and (line and line[0].islower() or (merged[-1] and merged[-1].endswith('-'))):
            merged[-1] = merged[-1].rstrip('-') + line
        else:
            merged.append(line)
    return merged


def load_pool(pool_dir, field):
    path = Path(pool_dir) / f"{field}_pool_embs.npz"
    npz = np.load(path)
    return npz["lines"].astype(str), npz["embeddings"].astype(np.float32)


def load_labels(labels_path):
    df = pd.read_csv(labels_path, encoding="utf-8-sig")
    df["eid"] = df["eid"].astype(str)
    df["pid"] = df["pid"].astype(str)
    cols = [c for c in DIAG_COLS if c in df.columns]
    patient_labels = df.groupby("pid")[cols].max()
    eid_to_pid = dict(zip(df["eid"], df["pid"]))
    return eid_to_pid, patient_labels, cols


def build_line_to_studies(h5_path):
    """Scan H5 → {merged_line: [study_ids]}."""
    line_to_studies = defaultdict(set)
    with h5py.File(h5_path, "r") as f:
        for sid in f.keys():
            raw = f[sid][()]
            lines = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]
            lines = merge_soft_wraps(lines)
            for line in lines:
                line_to_studies[line].add(sid)
    return line_to_studies


def get_positive_mask(lines, line_to_studies, eid_to_pid, patient_labels, col):
    """Return boolean mask: True if line appears in any positive study."""
    mask = np.zeros(len(lines), dtype=bool)
    for i, line in enumerate(lines):
        for sid in line_to_studies.get(line, []):
            pid = eid_to_pid.get(sid)
            if pid and pid in patient_labels.index:
                if patient_labels.loc[pid, col] == 1:
                    mask[i] = True
                    break
    return mask


def farthest_point_sample(coords, k, seed=42):
    """Greedy farthest point selection → k indices."""
    rng = np.random.RandomState(seed)
    n = len(coords)
    selected = [rng.randint(n)]
    dists = np.full(n, np.inf)
    for _ in range(k - 1):
        last = coords[selected[-1]]
        dists = np.minimum(dists, np.linalg.norm(coords - last, axis=1))
        selected.append(np.argmax(dists))
    return selected


def plot_umap_all(coords, path):
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, alpha=0.35, c="steelblue")
    ax.set_title(f"All Lines (n={len(coords):,})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_landmarks(coords, lines, indices, title, path):
    k = len(indices)
    cmap = cm.get_cmap("tab20", k)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, alpha=0.35, c="steelblue")
    
    for i, idx in enumerate(indices):
        color = cmap(i)
        label = lines[idx][:60] + "..." if len(lines[idx]) > 60 else lines[idx]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=80, c=[color],
                   edgecolors="k", linewidths=0.8, zorder=5, label=label)
    
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
              markerscale=1, frameon=True, title="Landmark Lines")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def eval_field(field, pool_dir, h5_dir, eid_to_pid, patient_labels, diag_cols,
               output_dir, n_landmarks):
    out = Path(output_dir) / field
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== {field} ===", flush=True)
    
    # Load pool
    lines, embs = load_pool(pool_dir, field)
    print(f"Loaded {len(lines):,} lines, {embs.shape[1]}d", flush=True)
    
    # Build line → studies mapping
    h5_path = Path(h5_dir) / f"{field}.h5"
    print(f"Scanning {h5_path.name}...", flush=True)
    line_to_studies = build_line_to_studies(h5_path)
    print(f"  {len(line_to_studies):,} unique lines in H5")
    
    # Fit UMAP once
    print("Fitting UMAP...", flush=True)
    coords = UMAP(n_neighbors=10, min_dist=0.1, metric="cosine", random_state=42).fit_transform(embs)
    
    # All lines
    plot_umap_all(coords, out / "umap_all.png")
    landmarks = farthest_point_sample(coords, n_landmarks)
    plot_landmarks(coords, lines, landmarks, f"{field} — All Lines", out / "umap_landmarks_all.png")
    
    # Per-diagnosis subgroups
    for col in diag_cols:
        mask = get_positive_mask(lines, line_to_studies, eid_to_pid, patient_labels, col)
        n_pos = mask.sum()
        if n_pos < n_landmarks:
            print(f"  Skipping {col}: only {n_pos} positive lines")
            continue
        
        sub_coords = coords[mask]
        sub_lines = lines[mask]
        sub_landmarks = farthest_point_sample(sub_coords, n_landmarks)
        plot_landmarks(sub_coords, sub_lines, sub_landmarks,
                       f"{field} — {col} (n={n_pos:,})",
                       out / f"umap_landmarks_{col.lower()}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_dir", required=True, help="Dir with {field}_pool_embs.npz files")
    p.add_argument("--h5_dir", required=True, help="line_tokenizer/data/ with per-field H5s")
    p.add_argument("--labels", required=True, help="Echo_Labels_SG_Fyler CSV")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_landmarks", type=int, default=15)
    p.add_argument("--fields", nargs="+", default=["study_findings", "summary", "history"])
    args = p.parse_args()
    
    eid_to_pid, patient_labels, diag_cols = load_labels(args.labels)
    print(f"Loaded {len(eid_to_pid):,} study→patient mappings, {len(diag_cols)} diagnoses")
    
    for field in args.fields:
        eval_field(field, args.pool_dir, args.h5_dir, eid_to_pid, patient_labels,
                   diag_cols, args.output_dir, args.n_landmarks)
    
    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
