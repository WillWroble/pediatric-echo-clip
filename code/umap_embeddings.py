"""UMAP visualization of PanEcho embeddings with sub-video clustering analysis."""

import os
import random
import numpy as np
import h5py
import umap
import hdbscan
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors

# --- Config ---
EMBED_DIR = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Embeddings/Embeddings/Internal")
OUT_DIR = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/analysis")
N_STUDY_SAMPLES = 5000
N_SUBVIDEO_STUDIES = 200  # fewer studies for sub-video UMAP (~200 * 50 = 10K points)
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_CLUSTER = 50
N_HIGHLIGHT_STUDIES = 5
KNN_INTRA = 5
KNN_INTER = 1
SEED = 42


def load_study_embedding(path):
    """Mean-pool frames per sub-video, mean-pool sub-videos per study -> (768,)"""
    try:
        with h5py.File(path, "r") as f:
            vecs = []
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith("/emb"):
                    vecs.append(obj[:].mean(axis=0))
            f.visititems(collect)
            if not vecs:
                return None, 0
            stacked = np.stack(vecs)
            return stacked.mean(axis=0), len(vecs)
    except Exception:
        return None, 0


def load_subvideo_embeddings(path, study_id):
    """Mean-pool frames per sub-video, return list of (study_id, subvideo_name, vec)."""
    rows = []
    try:
        with h5py.File(path, "r") as f:
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith("/emb"):
                    rows.append((study_id, name, obj[:].mean(axis=0)))
            f.visititems(collect)
    except Exception:
        pass
    return rows


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def compute_similarity_stats(subvideo_data):
    """Compute intra-study vs inter-study cosine similarity."""
    by_study = defaultdict(list)
    for sid, _, vec in subvideo_data:
        by_study[sid].append(vec)

    # intra-study: avg cosine sim between all pairs within each study
    intra_sims = []
    for vecs in by_study.values():
        if len(vecs) < 2:
            continue
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                intra_sims.append(cosine_sim(vecs[i], vecs[j]))

    # inter-study: sample pairs from different studies
    study_ids = list(by_study.keys())
    inter_sims = []
    rng = random.Random(SEED)
    for _ in range(min(len(intra_sims), 50000)):
        s1, s2 = rng.sample(study_ids, 2)
        v1 = rng.choice(by_study[s1])
        v2 = rng.choice(by_study[s2])
        inter_sims.append(cosine_sim(v1, v2))

    return np.array(intra_sims), np.array(inter_sims)


def nearest_neighbor_analysis(all_rows, out_dir, ks=[1, 5, 10, 20]):
    """Test connected-neighbor theory: are a sub-video's KNN disproportionately same-study?"""
    study_ids = np.array([r[0] for r in all_rows])
    vectors = np.stack([r[2] for r in all_rows])

    max_k = max(ks)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine")
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    # drop self-neighbor (column 0)
    neighbor_indices = indices[:, 1:]
    neighbor_dists = distances[:, 1:]

    # expected same-study rate by chance (weighted by study size)
    counts = Counter(study_ids)
    n = len(study_ids)
    expected = np.mean([(c - 1) / (n - 1) for c in counts.values() for _ in range(c)])

    print("\nNearest-neighbor same-study enrichment:")
    print(f"  Expected by chance: {expected:.4f}")
    results = {}
    for k in ks:
        same = 0
        total = 0
        for i in range(n):
            neighbors = neighbor_indices[i, :k]
            same += (study_ids[neighbors] == study_ids[i]).sum()
            total += k
        actual = same / total
        enrichment = actual / expected if expected > 0 else float("inf")
        results[k] = {"actual": actual, "enrichment": enrichment}
        print(f"  k={k:>2}: same-study={actual:.4f}  enrichment={enrichment:.1f}x")

    # per-study: fraction of each video's NN that are same-study (at k=10)
    k = 10
    per_video_rates = []
    for i in range(n):
        neighbors = neighbor_indices[i, :k]
        rate = (study_ids[neighbors] == study_ids[i]).mean()
        per_video_rates.append(rate)
    per_video_rates = np.array(per_video_rates)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(per_video_rates, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(expected, color="red", linestyle="--", label=f"chance={expected:.4f}")
    ax.axvline(per_video_rates.mean(), color="blue", linestyle="--", label=f"actual mean={per_video_rates.mean():.4f}")
    ax.set_xlabel(f"Fraction of k={k} nearest neighbors from same study")
    ax.set_ylabel("Count (sub-videos)")
    ax.set_title(f"Connected-Neighbor Test: Same-Study KNN Rate (n={n})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "knn_same_study.png", dpi=200)
    print("Saved → knn_same_study.png")

    return results


def connected_neighbor_viz(all_rows, coords, out_dir):
    """Visualize connected-neighbor chains for selected studies on the full UMAP."""
    study_ids = np.array([r[0] for r in all_rows])
    vectors = np.stack([r[2] for r in all_rows])
    unique_studies = list(set(study_ids))

    # pick studies spread across UMAP space by selecting from different spatial regions
    study_centroids = {}
    for sid in unique_studies:
        mask = study_ids == sid
        study_centroids[sid] = coords[mask].mean(axis=0)

    # k-means-style selection: pick studies whose centroids are spread out
    selected = [random.choice(unique_studies)]
    for _ in range(N_HIGHLIGHT_STUDIES - 1):
        best_sid, best_dist = None, -1
        for sid in unique_studies:
            if sid in selected:
                continue
            min_dist = min(np.linalg.norm(study_centroids[sid] - study_centroids[s]) for s in selected)
            if min_dist > best_dist:
                best_sid, best_dist = sid, min_dist
        selected.append(best_sid)

    print(f"\nConnected-neighbor viz: selected studies {selected}")
    for sid in selected:
        n = (study_ids == sid).sum()
        print(f"  {sid}: {n} sub-videos")

    # KNN for edges
    max_k = max(KNN_INTRA, KNN_INTER)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine")
    nn.fit(vectors)
    _, indices = nn.kneighbors(vectors)
    neighbor_indices = indices[:, 1:]  # drop self

    highlight_set = set(selected)
    highlight_mask = np.array([s in highlight_set for s in study_ids])
    highlight_indices = np.where(highlight_mask)[0]

    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]
    study_color = {sid: colors[i] for i, sid in enumerate(selected)}

    fig, ax = plt.subplots(figsize=(14, 11))

    # gray background: all points
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=2, alpha=0.15, zorder=1)

    # inter-study edges (gray): for highlighted videos, k=1 nearest neighbor if from different study
    for i in highlight_indices:
        for j_idx in range(KNN_INTER):
            j = neighbor_indices[i, j_idx]
            if study_ids[j] != study_ids[i]:
                ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                        color="gray", alpha=0.15, linewidth=0.5, zorder=2)
                break  # only first inter-study neighbor

    # intra-study edges (colored): for highlighted videos, k=5 nearest same-study neighbors
    for sid in selected:
        mask = study_ids == sid
        sid_indices = np.where(mask)[0]
        sid_set = set(sid_indices)
        for i in sid_indices:
            drawn = 0
            for j in neighbor_indices[i]:
                if j in sid_set and drawn < KNN_INTRA:
                    ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                            color=study_color[sid], alpha=0.4, linewidth=0.8, zorder=3)
                    drawn += 1

    # highlighted study dots on top
    for sid in selected:
        mask = study_ids == sid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=study_color[sid], s=15, alpha=0.9, zorder=4,
                   edgecolors="black", linewidths=0.3, label=f"Study {sid}")

    ax.set_title(f"Connected-Neighbor Chains — {N_HIGHLIGHT_STUDIES} studies "
                 f"(intra k={KNN_INTRA}, inter k={KNN_INTER})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper left", fontsize=8, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(out_dir / "connected_neighbors.png", dpi=200)
    print("Saved → connected_neighbors.png")


def run_study_level(files, out_dir):
    """Study-level UMAP + HDBSCAN."""
    print("=" * 60)
    print("STUDY-LEVEL ANALYSIS")
    print("=" * 60)

    sampled = random.sample(files, min(N_STUDY_SAMPLES, len(files)))
    study_ids, vectors, subvideo_counts = [], [], []

    for i, fname in enumerate(sampled):
        vec, n_clips = load_study_embedding(EMBED_DIR / fname)
        if vec is not None:
            study_ids.append(fname.replace("_trim_embed.hdf5", ""))
            vectors.append(vec)
            subvideo_counts.append(n_clips)
        if (i + 1) % 500 == 0:
            print(f"  loaded {i + 1}/{len(sampled)} ({len(vectors)} valid)")

    X = np.stack(vectors)
    counts = np.array(subvideo_counts)
    print(f"Embedding matrix: {X.shape}")

    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        metric="cosine", random_state=SEED,
    )
    coords = reducer.fit_transform(X)

    # HDBSCAN
    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER)
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    # plot
    fig, ax = plt.subplots(figsize=(12, 10))
    noise_mask = labels == -1
    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
               c="lightgray", s=3, alpha=0.3, label="noise")
    scatter = ax.scatter(coords[~noise_mask, 0], coords[~noise_mask, 1],
                         c=labels[~noise_mask], cmap="tab20", s=5, alpha=0.6)
    ax.set_title(f"PanEcho Internal Embeddings — {n_clusters} clusters ({len(vectors)} studies)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(out_dir / "umap_study_level.png", dpi=200)
    print("Saved → umap_study_level.png")

    # cluster composition by sub-video count
    print(f"\nCluster composition by sub-video count:")
    print(f"{'Cluster':>8} {'N':>6} {'Mean clips':>11} {'Median':>7} {'Min':>5} {'Max':>5}")
    for c in sorted(set(labels)):
        mask = labels == c
        cc = counts[mask]
        name = "noise" if c == -1 else str(c)
        print(f"{name:>8} {mask.sum():>6} {cc.mean():>11.1f} {np.median(cc):>7.0f} {cc.min():>5} {cc.max():>5}")

    # save
    np.savez(out_dir / "umap_study_results.npz",
             study_ids=np.array(study_ids), coords=coords,
             labels=labels, vectors=X, subvideo_counts=counts)
    print("Saved → umap_study_results.npz")


def run_subvideo_level(files, out_dir):
    """Sub-video-level UMAP: how do videos within studies relate?"""
    print("\n" + "=" * 60)
    print("SUB-VIDEO-LEVEL ANALYSIS")
    print("=" * 60)

    sampled = random.sample(files, min(N_SUBVIDEO_STUDIES, len(files)))
    all_rows = []

    for i, fname in enumerate(sampled):
        sid = fname.replace("_trim_embed.hdf5", "")
        rows = load_subvideo_embeddings(EMBED_DIR / fname, sid)
        all_rows.extend(rows)
        if (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(sampled)} studies ({len(all_rows)} sub-videos)")

    study_ids = [r[0] for r in all_rows]
    vectors = np.stack([r[2] for r in all_rows])
    unique_studies = list(set(study_ids))
    study_to_idx = {s: i for i, s in enumerate(unique_studies)}
    study_labels = np.array([study_to_idx[s] for s in study_ids])
    print(f"Sub-video matrix: {vectors.shape} from {len(unique_studies)} studies")

    # intra vs inter similarity
    print("\nComputing intra vs inter-study similarity...")
    intra, inter = compute_similarity_stats(all_rows)
    print(f"  Intra-study cosine sim:  mean={intra.mean():.4f}  std={intra.std():.4f}  median={np.median(intra):.4f}")
    print(f"  Inter-study cosine sim:  mean={inter.mean():.4f}  std={inter.std():.4f}  median={np.median(inter):.4f}")
    print(f"  Separation ratio (intra/inter): {intra.mean() / inter.mean():.3f}")

    # nearest-neighbor same-study enrichment
    nearest_neighbor_analysis(all_rows, out_dir)

    # similarity histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(intra, bins=100, alpha=0.6, label=f"Intra-study (μ={intra.mean():.3f})", density=True)
    ax.hist(inter, bins=100, alpha=0.6, label=f"Inter-study (μ={inter.mean():.3f})", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Intra-study vs Inter-study Sub-video Similarity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "similarity_histogram.png", dpi=200)
    print("Saved → similarity_histogram.png")

    # sub-video count distribution
    clips_per_study = defaultdict(int)
    for sid in study_ids:
        clips_per_study[sid] += 1
    clip_counts = np.array(list(clips_per_study.values()))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(clip_counts, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Sub-videos per Study")
    ax.set_ylabel("Count")
    ax.set_title(f"Sub-video Count Distribution (n={len(clip_counts)}, μ={clip_counts.mean():.1f}, median={np.median(clip_counts):.0f})")
    ax.axvline(clip_counts.mean(), color="red", linestyle="--", label=f"mean={clip_counts.mean():.1f}")
    ax.axvline(np.median(clip_counts), color="orange", linestyle="--", label=f"median={np.median(clip_counts):.0f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "subvideo_count_dist.png", dpi=200)
    print("Saved → subvideo_count_dist.png")

    # UMAP colored by study
    print("Running sub-video UMAP...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        metric="cosine", random_state=SEED,
    )
    coords = reducer.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1],
               c=study_labels, cmap="tab20", s=3, alpha=0.4)
    ax.set_title(f"Sub-video Embeddings by Study ({vectors.shape[0]} clips, {len(unique_studies)} studies)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_dir / "umap_subvideo_by_study.png", dpi=200)
    print("Saved → umap_subvideo_by_study.png")

    # UMAP with HDBSCAN (view-type discovery)
    print("Running sub-video HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER)
    hdb_labels = clusterer.fit_predict(coords)
    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    print(f"Found {n_clusters} sub-video clusters (potential view types)")

    fig, ax = plt.subplots(figsize=(12, 10))
    noise_mask = hdb_labels == -1
    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
               c="lightgray", s=3, alpha=0.2)
    scatter = ax.scatter(coords[~noise_mask, 0], coords[~noise_mask, 1],
                         c=hdb_labels[~noise_mask], cmap="tab20", s=3, alpha=0.5)
    ax.set_title(f"Sub-video Embeddings — {n_clusters} HDBSCAN clusters (potential view types)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(out_dir / "umap_subvideo_hdbscan.png", dpi=200)
    print("Saved → umap_subvideo_hdbscan.png")

    # connected-neighbor chain visualization
    connected_neighbor_viz(all_rows, coords, out_dir)

    # save
    np.savez(out_dir / "umap_subvideo_results.npz",
             study_ids=np.array(study_ids), coords=coords,
             hdbscan_labels=hdb_labels, study_labels=study_labels, vectors=vectors)
    print("Saved → umap_subvideo_results.npz")


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(EMBED_DIR) if f.endswith(".hdf5")]
    print(f"Found {len(files)} HDF5 files\n")

    run_study_level(files, OUT_DIR)
    run_subvideo_level(files, OUT_DIR)

    print("\n" + "=" * 60)
    print(f"All outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
