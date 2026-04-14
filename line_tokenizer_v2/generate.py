"""Generate heatmaps via video-level cross-attention.

For each study: centroids attend over videos → dot product → sigmoid → hotspots.

Usage:
    python -u generate.py \
        --codebook results/v1/codebook.npz \
        --checkpoint results/v1/latest.pt \
        --video_embeddings /path/to/infonce_768_all.npz \
        --h5_path /path/to/study_findings.h5 \
        --line_embeddings results/v1/line_embeddings.npz \
        --output results/v1/heatmaps.json
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def find_hotspots(scores, sim_matrix, threshold=0.3, linkage_cutoff=0.85):
    """Group high-scoring tokens into hotspots by cosine proximity."""
    active = np.where(scores > threshold)[0]
    if len(active) == 0:
        return []
    if len(active) == 1:
        return [[(int(active[0]), float(scores[active[0]]))]]

    sub_sim = sim_matrix[np.ix_(active, active)]
    adjacency = (sub_sim >= linkage_cutoff).astype(np.int32)
    np.fill_diagonal(adjacency, 0)

    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)

    hotspots = []
    for c in range(n_components):
        members = active[labels == c]
        ranked = sorted([(int(m), float(scores[m])) for m in members], key=lambda x: -x[1])
        hotspots.append(ranked)

    hotspots.sort(key=lambda h: -h[0][1])
    return hotspots


def assign_lines_to_clusters(line_embs_path, codebook_path, batch_size=8192):
    """Assign every encoded line to its nearest codebook centroid."""
    cb = np.load(codebook_path, allow_pickle=True)
    centroids = cb["centroids"].astype(np.float32)
    cluster_ids = cb["cluster_ids"]
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    le = np.load(line_embs_path, allow_pickle=True)
    lines = le["lines"].astype(str).tolist()
    embs = le["embeddings"].astype(np.float32)

    mapping = {}
    for i in range(0, len(lines), batch_size):
        batch_embs = embs[i:i + batch_size]
        batch_norm = batch_embs / (np.linalg.norm(batch_embs, axis=1, keepdims=True) + 1e-8)
        sims = batch_norm @ centroids_norm.T
        idxs = sims.argmax(axis=1)
        for j, line in enumerate(lines[i:i + batch_size]):
            mapping[line] = int(cluster_ids[idxs[j]])

    print(f"Assigned {len(mapping):,} lines to {len(cluster_ids)} clusters", flush=True)
    return mapping


def load_video_embeddings_by_study(npz_path):
    """Load video-level embeddings grouped by study."""
    data = np.load(npz_path)
    embs = data["embeddings"]
    sids = data["study_ids"].astype(str)
    by_study = {}
    for emb, sid in zip(embs, sids):
        by_study.setdefault(sid, []).append(emb)
    return {k: np.stack(v, dtype=np.float32) for k, v in by_study.items()}


def load_study_lines(h5_path, study_ids, line_filters=None):
    """Load reference lines for each study."""
    patterns = []
    if line_filters:
        patterns = [re.compile(l.strip(), re.IGNORECASE)
                    for l in open(line_filters) if l.strip() and not l.startswith("#")]

    study_lines = {}
    ids_set = set(study_ids)
    with h5py.File(h5_path, "r") as f:
        for sid_raw in f.keys():
            sid = str(int(float(sid_raw)))
            if sid not in ids_set:
                continue
            lines = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f[sid_raw][()]]
            if patterns:
                lines = [l for l in lines if not any(p.search(l) for p in patterns)]
            study_lines[sid] = lines
    return study_lines


@torch.no_grad()
def generate_heatmaps(study_ids, video_embs, centroids, cluster_ids, rep_texts,
                      sim_matrix, W_Q, W_K, scale, study_lines, line_to_cluster,
                      cid_to_idx, device, threshold=0.3, linkage_cutoff=0.85, top_k=10):
    results = []
    centroids_t = torch.from_numpy(centroids).to(device)  # (K, 768)

    for sid in study_ids:
        videos = torch.from_numpy(video_embs[sid]).to(device)  # (V, 768)

        # Cross-attention: each centroid attends over videos
        Q = W_Q(centroids_t)        # (K, 768)
        K = W_K(videos)             # (V, 768)
        attn = (Q @ K.T) * scale    # (K, V)
        weights = attn.softmax(dim=-1)
        attended = weights @ videos  # (K, 768)

        # Dot product in video space
        #logits = (F.normalize(centroids_t, dim=-1) * F.normalize(attended, dim=-1)).sum(dim=-1).cpu().numpy()
        logits = (centroids_t * attended).sum(dim=-1).cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-logits))

        hotspots = find_hotspots(scores, sim_matrix, threshold, linkage_cutoff)

        hotspot_data = []
        for h_members in hotspots:
            top = h_members[:top_k]
            hotspot_data.append([
                {"cluster_id": int(cluster_ids[idx]), "score": round(score, 4),
                 "text": str(rep_texts[idx])}
                for idx, score in top
            ])

        ref_lines = study_lines.get(sid, [])
        ref_clusters = []
        for line in ref_lines:
            cid = line_to_cluster.get(line)
            if cid is not None and cid in cid_to_idx:
                ref_clusters.append({
                    "cluster_id": int(cid),
                    "text": line,
                    "score": round(float(scores[cid_to_idx[cid]]), 4),
                })

        n_unmapped = sum(1 for line in ref_lines
                         if line_to_cluster.get(line) is None or line_to_cluster.get(line) not in cid_to_idx)
        top_scores = np.sort(scores)[::-1]

        results.append({
            "study_id": sid,
            "n_hotspots": len(hotspot_data),
            "n_active": int((scores > threshold).sum()),
            "n_unmapped": n_unmapped,
            "top_5_scores": [round(float(s), 4) for s in top_scores[:5]],
            "hotspots": hotspot_data,
            "reference": ref_clusters,
        })

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codebook", required=True)
    p.add_argument("--checkpoint", required=True, help="CrossAttentionPool weights")
    p.add_argument("--video_embeddings", required=True)
    p.add_argument("--h5_path", required=True)
    p.add_argument("--line_embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--line_filters", default=None)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--linkage_cutoff", type=float, default=0.85)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # Load codebook
    cb = np.load(args.codebook, allow_pickle=True)
    cluster_ids = cb["cluster_ids"]
    centroids = cb["centroids"].astype(np.float32)
    rep_texts = cb["labels"].astype(str)
    cid_to_idx = {int(c): i for i, c in enumerate(cluster_ids)}

    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sim_matrix = centroids_norm @ centroids_norm.T
    print(f"Loaded codebook: {len(cluster_ids)} clusters", flush=True)

    # Load CrossAttentionPool weights
    from model import CrossAttentionPool
    pool = CrossAttentionPool(dim=768).to(device)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    pool.load_state_dict(ckpt["attn_pool"])
    pool.eval()
    print("Loaded CrossAttentionPool checkpoint", flush=True)

    # Load video embeddings
    video_embs = load_video_embeddings_by_study(args.video_embeddings)
    ids = list(video_embs.keys())
    print(f"Loaded {len(ids):,} studies with video embeddings", flush=True)

    # Filter by manifest
    if args.manifest:
        manifest = set(str(int(float(x)))
                       for x in Path(args.manifest).read_text().strip().splitlines())
        ids = [s for s in ids if s in manifest]
        print(f"Manifest filter: {len(ids):,} studies", flush=True)

    # Subsample
    rng = np.random.default_rng(args.seed)
    if args.n_samples and len(ids) > args.n_samples:
        ids = list(rng.choice(ids, args.n_samples, replace=False))
    print(f"Generating heatmaps for {len(ids)} studies", flush=True)

    # Line → cluster mapping
    line_to_cluster = assign_lines_to_clusters(args.line_embeddings, args.codebook)

    # Load reference lines
    study_lines = load_study_lines(args.h5_path, ids, args.line_filters)

    # Generate
    results = generate_heatmaps(
        ids, video_embs, centroids, cluster_ids, rep_texts, sim_matrix,
        pool.W_Q, pool.W_K, pool.scale, study_lines, line_to_cluster,
        cid_to_idx, device, args.threshold, args.linkage_cutoff, args.top_k,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    total_unmapped = sum(r["n_unmapped"] for r in results)
    total_ref = sum(len(r["reference"]) + r["n_unmapped"] for r in results)
    print(f"Saved {len(results)} heatmaps to {args.output}", flush=True)
    print(f"Unmapped: {total_unmapped}/{total_ref} ({100*total_unmapped/max(total_ref,1):.1f}%)", flush=True)

    for r in results[:3]:
        print(f"\n{'='*60}")
        print(f"Study {r['study_id']}: {r['n_hotspots']} hotspots, {r['n_active']} active")
        for i, hs in enumerate(r["hotspots"][:3]):
            print(f"\n  Hotspot {i+1}:")
            for t in hs[:3]:
                print(f"    [{t['score']:.3f}] {t['text'][:80]}")


if __name__ == "__main__":
    main()
