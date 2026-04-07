"""Generate clinical pathology heatmaps from LineDecoder.

For each study: predict 10K sigmoid scores, group high-scoring tokens
into hotspots by cosine proximity, output top clusters per hotspot.

Usage:
    python -u generate.py \
        --checkpoint results/v1/best.pt \
        --embeddings /lab-share/.../embeddings_video.npz \
        --codebook /lab-share/.../codebook.npz \
        --h5_path /lab-share/.../study_findings.h5 \
        --line_embeddings /lab-share/.../line_embeddings.npz \
        --output results/v1/heatmaps.json \
        --n_samples 200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from model import LineDecoder
from dataset import assign_lines_to_clusters


def find_hotspots(scores, sim_matrix, threshold=0.3, linkage_cutoff=0.85):
    """Group high-scoring tokens into hotspots by cosine proximity.

    Returns list of hotspots, each a list of (token_idx, score) sorted by score desc.
    """
    active = np.where(scores > threshold)[0]
    if len(active) == 0:
        return []
    if len(active) == 1:
        return [[(int(active[0]), float(scores[active[0]]))]]

    # Pairwise cosine between active tokens
    sub_sim = sim_matrix[np.ix_(active, active)]
    adjacency = (sub_sim >= linkage_cutoff).astype(np.int32)
    np.fill_diagonal(adjacency, 0)

    n_components, component_labels = connected_components(
        csr_matrix(adjacency), directed=False)

    hotspots = []
    for c in range(n_components):
        members = active[component_labels == c]
        ranked = sorted([(int(m), float(scores[m])) for m in members],
                        key=lambda x: -x[1])
        hotspots.append(ranked)

    hotspots.sort(key=lambda h: -h[0][1])
    return hotspots


@torch.no_grad()
def generate_heatmaps(model, study_ids, study_embs, codebook_path,
                      sim_matrix, rep_texts, cluster_ids,
                      study_lines, line_to_cluster, cid_to_idx,
                      device, threshold=0.3, linkage_cutoff=0.85,
                      top_k_per_hotspot=10):
    model.eval()
    results = []

    for sid in study_ids:
        emb = torch.from_numpy(study_embs[sid]).unsqueeze(0).to(device)
        logits = model(emb).squeeze(0).cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

        hotspots = find_hotspots(scores, sim_matrix, threshold, linkage_cutoff)

        # Format hotspots
        hotspot_data = []
        for h_members in hotspots:
            top = h_members[:top_k_per_hotspot]
            hotspot_data.append([
                {"cluster_id": int(cluster_ids[idx]),
                 "score": round(score, 4),
                 "text": str(rep_texts[idx])}
                for idx, score in top
            ])

        # Reference: actual lines and their cluster assignments
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

        # Global stats
        n_unmapped = sum(1 for line in ref_lines if line_to_cluster.get(line) is None or line_to_cluster.get(line) not in cid_to_idx)
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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", required=True, help="study embeddings npz")
    p.add_argument("--manifest", default=None)
    p.add_argument("--codebook", required=True)
    p.add_argument("--h5_path", required=True, help="preprocess H5 for reference lines")
    p.add_argument("--line_embeddings", required=True, help="encode.py output for line→cluster mapping")
    p.add_argument("--line_filters", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--linkage_cutoff", type=float, default=0.85)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # Load codebook
    cb = np.load(args.codebook, allow_pickle=True)
    cluster_ids = cb["cluster_ids"]
    centroids = cb["centroids"].astype(np.float32)
    rep_texts = cb["labels"].astype(str)
    vocab_size = len(cluster_ids)
    cid_to_idx = {int(c): i for i, c in enumerate(cluster_ids)}

    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sim_matrix = centroids_norm @ centroids_norm.T

    # Load model
    # Load model
    config_path = Path(args.checkpoint).parent / "config.json"
    dropout = 0.0
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
            dropout = cfg.get("dropout", 0.0)
    model = LineDecoder(vocab_size=vocab_size, dropout=dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location=device), strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    # Load study embeddings
    npz = np.load(args.embeddings)
    if args.split == "val":
        arr = npz["val"].astype(np.float32)
        ids = [str(int(float(x))) for x in npz["val_ids"]]
    else:
        arr = npz["train"].astype(np.float32)
        ids = [str(int(float(x))) for x in npz["train_ids"]]
    study_embs = {sid: arr[i] for i, sid in enumerate(ids)}

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
    print("Assigning lines to clusters...", flush=True)
    line_to_cluster = assign_lines_to_clusters(args.line_embeddings, args.codebook)

    # Load reference lines
    import re
    import h5py
    patterns = []
    if args.line_filters:
        patterns = [re.compile(l.strip(), re.IGNORECASE)
                    for l in open(args.line_filters)
                    if l.strip() and not l.startswith("#")]

    study_lines = {}
    ids_set = set(ids)
    with h5py.File(args.h5_path, "r") as f:
        for sid_raw in f.keys():
            sid = str(int(float(sid_raw)))
            if sid not in ids_set:
                continue
            lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                     for x in f[sid_raw][()]]
            if patterns:
                lines = [l for l in lines if not any(p.search(l) for p in patterns)]
            study_lines[sid] = lines

    # Generate
    results = generate_heatmaps(
        model, ids, study_embs, args.codebook,
        sim_matrix, rep_texts, cluster_ids,
        study_lines, line_to_cluster, cid_to_idx,
        device, args.threshold, args.linkage_cutoff, args.top_k,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    total_unmapped = sum(r["n_unmapped"] for r in results)
    total_ref = sum(len(r["reference"]) + r["n_unmapped"] for r in results)
    print(f"Saved {len(results)} heatmaps to {args.output}", flush=True)
    print(f"Unmapped reference lines: {total_unmapped}/{total_ref} ({100*total_unmapped/max(total_ref,1):.1f}%)", flush=True)

    # Print a few examples
    for r in results[:3]:
        print(f"\n{'='*60}")
        print(f"Study {r['study_id']}: {r['n_hotspots']} hotspots, "
              f"{r['n_active']} active tokens")
        for i, hs in enumerate(r["hotspots"][:5]):
            print(f"\n  Hotspot {i+1}:")
            for t in hs[:5]:
                print(f"    [{t['score']:.3f}] {t['text'][:80]}")
        print(f"\n  Reference ({len(r['reference'])} lines):")
        for ref in r["reference"][:5]:
            print(f"    [{ref['score']:.3f}] {ref['text'][:80]}")


if __name__ == "__main__":
    main()
