"""Visualize per-study heatmaps as UMAPs with score, frequency, and co-occurrence coloring."""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix

from model import LineEncoder, CrossAttentionPool


def merge_soft_wraps(lines):
    if not lines:
        return []
    merged = [lines[0]]
    for line in lines[1:]:
        prev = merged[-1]
        if line[0].islower() or prev.endswith('-'):
            sep = '' if prev.endswith('-') else ' '
            merged[-1] = prev.rstrip('-') + sep + line
        else:
            merged.append(line)
    return merged


def build_line_to_studies(h5_path, manifest):
    """Map each line to set of study_ids where it appears."""
    line_to_studies = defaultdict(set)
    with h5py.File(h5_path, "r") as f:
        for sid_raw in f.keys():
            sid = str(int(float(sid_raw)))
            if sid not in manifest:
                continue
            lines = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f[sid_raw][()]]
            lines = merge_soft_wraps(lines)
            for line in lines:
                line_to_studies[line].add(sid)
    return line_to_studies


def mutual_knn_hotspots(coords, scores, k=10, score_threshold=0.5):
    """
    Cluster high-scoring points via mutual KNN + connected components.
    Returns list of (hotspot_indices, peak_idx) tuples.
    """
    high_mask = scores >= score_threshold
    high_idx = np.where(high_mask)[0]
    
    if len(high_idx) < 2:
        if len(high_idx) == 1:
            return [(high_idx, high_idx[0])]
        return []
    
    high_coords = coords[high_idx]
    k_use = min(k, len(high_idx) - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean").fit(high_coords)
    _, indices = nbrs.kneighbors(high_coords)
    
    # Build mutual graph
    n = len(high_idx)
    adj = lil_matrix((n, n), dtype=bool)
    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            if i in indices[j, 1:]:
                adj[i, j] = True
                adj[j, i] = True
    
    n_components, labels = connected_components(adj, directed=False)
    
    hotspots = []
    for c in range(n_components):
        members = high_idx[labels == c]
        peak = members[scores[members].argmax()]
        hotspots.append((members, peak))
    
    return hotspots


def plot_umap_scored(coords, scores, lines, hotspots, title, path, top_n=10):
    """UMAP colored by score with hotspot landmarks."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="Reds", 
                    s=3, alpha=0.5, vmin=0, vmax=max(scores.max(), 1))
    plt.colorbar(sc, ax=ax, label="Score")
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(hotspots), 1)))
    for i, (members, peak) in enumerate(hotspots[:top_n]):
        ax.scatter(coords[peak, 0], coords[peak, 1], c=[colors[i]], s=100, 
                   edgecolors="black", linewidths=1, zorder=10)
        ax.annotate(f"{i+1}", (coords[peak, 0], coords[peak, 1]), fontsize=8, 
                    ha="center", va="bottom", weight="bold")
    
    legend_text = "\n".join([f"{i+1}. {lines[peak][:60]}" 
                             for i, (_, peak) in enumerate(hotspots[:top_n])])
    ax.text(1.02, 0.98, legend_text, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_umap_reference(coords, scores, lines, ref_indices, title, path, top_n=15):
    """UMAP colored by score with reference line landmarks."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="Reds", 
                    s=3, alpha=0.5, vmin=0, vmax=max(scores.max(), 1))
    plt.colorbar(sc, ax=ax, label="Score")
    
    # Sort reference by score
    ref_sorted = sorted(ref_indices, key=lambda i: scores[i], reverse=True)
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(ref_sorted[:top_n]), 1)))
    for i, idx in enumerate(ref_sorted[:top_n]):
        ax.scatter(coords[idx, 0], coords[idx, 1], c=[colors[i]], s=100, 
                   edgecolors="black", linewidths=1, zorder=10, marker="s")
        ax.annotate(f"{i+1}", (coords[idx, 0], coords[idx, 1]), fontsize=8, 
                    ha="center", va="bottom", weight="bold")
    
    legend_text = "\n".join([f"{i+1}. {lines[idx][:60]}" 
                             for i, idx in enumerate(ref_sorted[:top_n])])
    ax.text(1.02, 0.98, legend_text, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_umap_colored(coords, values, title, path, cmap="viridis", label="Value"):
    """Generic UMAP with color gradient."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=3, alpha=0.5)
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_npz", required=True)
    p.add_argument("--video_embeddings", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5_path", required=True)
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--study_manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--knn", type=int, default=10)
    p.add_argument("--score_threshold", type=float, default=0.5)
    p.add_argument("--max_videos", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load pool
    pool = np.load(args.pool_npz, allow_pickle=True)
    pool_lines = pool["lines"]
    pool_embs = pool["embeddings"]
    print(f"Pool: {len(pool_lines)} lines", flush=True)
    
    line_to_idx = {line: i for i, line in enumerate(pool_lines)}

    # Load manifests
    train_manifest = set()
    for x in Path(args.train_manifest).read_text().strip().splitlines():
        train_manifest.add(str(int(float(x))))
    
    study_sids = []
    for x in Path(args.study_manifest).read_text().strip().splitlines():
        study_sids.append(str(int(float(x))))
    print(f"Eval studies: {len(study_sids)}", flush=True)

    # Load videos
    data = np.load(args.video_embeddings)
    embs, sids = data["embeddings"], data["study_ids"].astype(str)
    videos_by_study = {}
    for emb, sid in zip(embs, sids):
        sid = str(int(float(sid)))
        videos_by_study.setdefault(sid, []).append(emb)
    videos_by_study = {k: np.stack(v).astype(np.float32) for k, v in videos_by_study.items()}

    # Build line_to_studies from train manifest
    print("Building line_to_studies...", flush=True)
    line_to_studies = build_line_to_studies(args.h5_path, train_manifest)

    # Load reference lines per eval study
    ref_lines_by_study = {}
    with h5py.File(args.h5_path, "r") as f:
        for sid in study_sids:
            if sid in f.keys() or str(int(sid)) in f.keys():
                key = sid if sid in f.keys() else str(int(sid))
                lines = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f[key][()]]
                lines = merge_soft_wraps(lines)
                ref_lines_by_study[sid] = set(lines)

    # Compute line frequencies
    print("Computing frequencies...", flush=True)
    freq = np.array([len(line_to_studies.get(line, set())) for line in pool_lines], dtype=np.float32)

    # Load model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    encoder = LineEncoder().to(device)
    attn_pool = CrossAttentionPool().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    attn_pool.load_state_dict(ckpt["attn_pool"])
    encoder.eval()
    attn_pool.eval()

    # Encode pool lines
    print("Encoding pool lines...", flush=True)
    enc = tokenizer(list(pool_lines), padding="max_length", truncation=True, 
                    max_length=128, return_tensors="pt")
    pool_ids = enc["input_ids"].to(device)
    pool_masks = enc["attention_mask"].to(device)
    
    with torch.no_grad():
        batch_size = 256
        pool_line_embs = []
        for i in range(0, len(pool_lines), batch_size):
            emb = encoder(pool_ids[i:i+batch_size], pool_masks[i:i+batch_size])
            pool_line_embs.append(emb.cpu())
        pool_line_embs = torch.cat(pool_line_embs, dim=0).numpy()

    # Fit UMAP
    print("Fitting UMAP...", flush=True)
    umap = UMAP(n_neighbors=10, min_dist=0.1, metric="cosine", random_state=42)
    coords = umap.fit_transform(pool_line_embs)

    # Global frequency UMAP
    print("Plotting frequency UMAP...", flush=True)
    plot_umap_colored(coords, np.log1p(freq), "Line Frequency (log)", 
                      Path(args.output_dir) / "frequency_umap.png", 
                      cmap="viridis", label="log(1 + count)")

    # Per-study UMAPs
    for sid in study_sids:
        print(f"Processing {sid}...", flush=True)
        out_dir = Path(args.output_dir) / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        if sid not in videos_by_study:
            print(f"  Skipping {sid}: no videos", flush=True)
            continue

        vids = videos_by_study[sid]
        if vids.shape[0] > args.max_videos:
            idx = np.random.choice(vids.shape[0], args.max_videos, replace=False)
            vids = vids[idx]
        vids_t = torch.from_numpy(vids).unsqueeze(0).to(device)
        mask_t = torch.ones(1, vids_t.shape[1], device=device)

        # Score all pool lines
        with torch.no_grad():
            scores = []
            for i in range(0, len(pool_lines), 256):
                lines_t = torch.from_numpy(pool_line_embs[i:i+256]).unsqueeze(0).to(device)
                vids_batch = vids_t.expand(1, -1, -1)
                mask_batch = mask_t.expand(1, -1)
                attended = attn_pool(lines_t, vids_batch, mask_batch)
                logits = (lines_t * attended).sum(dim=-1).squeeze(0)
                scores.append(logits.cpu().numpy())
            scores = np.concatenate(scores)

        # Normalize scores for visualization
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Hotspot clustering
        hotspots = mutual_knn_hotspots(coords, scores_norm, k=args.knn, 
                                        score_threshold=args.score_threshold)
        hotspots = sorted(hotspots, key=lambda x: scores[x[1]], reverse=True)

        # UMAP 1: score colored with hotspot landmarks
        plot_umap_scored(coords, scores_norm, pool_lines, hotspots,
                         f"Study {sid} - Score Heatmap (Hotspots)", 
                         out_dir / "score_hotspots_umap.png")

        # UMAP 2: score colored with reference landmarks
        ref_lines = ref_lines_by_study.get(sid, set())
        ref_indices = [line_to_idx[line] for line in ref_lines if line in line_to_idx]
        
        if ref_indices:
            plot_umap_reference(coords, scores_norm, pool_lines, ref_indices,
                               f"Study {sid} - Score Heatmap (Reference)",
                               out_dir / "score_reference_umap.png")
        else:
            print(f"  No reference lines in pool for {sid}", flush=True)

        # UMAP 3: co-occurrence colored
        studies_with_ref = set()
        for line in ref_lines:
            studies_with_ref.update(line_to_studies.get(line, set()))
        
        cooccur = np.array([len(line_to_studies.get(line, set()) & studies_with_ref) 
                           for line in pool_lines], dtype=np.float32)
        
        plot_umap_colored(coords, np.log1p(cooccur), 
                          f"Study {sid} - Co-occurrence with Reference",
                          out_dir / "cooccurrence_umap.png",
                          cmap="Purples", label="log(1 + co-occurrence)")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
