"""Cluster line embeddings into a semantic codebook.

Two input modes:
  --npz: encode.py output (embeddings + lines arrays)
  --h5_dir + --manifest + --field: raw CLS from Line_Embeddings_v2

Usage (new):
    python -u cluster.py \
        --npz results/v3/line_embeddings.npz \
        --min_cluster_size 10 \
        --output_dir results/v3/clusters

Usage (legacy):
    python -u cluster.py \
        --h5_dir /lab-share/.../Line_Embeddings_v2 \
        --manifest manifests/train_50_nofetal.txt \
        --field study_findings \
        --min_cluster_size 10 \
        --output_dir results/legacy_clusters
"""

import argparse
import re
from pathlib import Path
from collections import Counter

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_from_npz(npz_path):
    """Load encode.py output. Returns (texts, embeddings)."""
    data = np.load(npz_path, allow_pickle=True)
    texts = data["lines"].astype(str).tolist()
    embs = data["embeddings"].astype(np.float32)
    print(f"Loaded {len(texts):,} lines from {npz_path}", flush=True)
    return texts, embs


def load_from_h5(h5_dir, manifest_path, field, ignore_patterns):
    """Load raw CLS embeddings from Line_Embeddings_v2 chunks."""
    manifest = set(str(int(float(x))) for x in
                   Path(manifest_path).read_text().strip().splitlines())
    print(f"Manifest: {len(manifest):,} studies")

    files = sorted(Path(h5_dir).glob("chunk_*.h5"))
    text_key = f"{field}_text"
    all_texts, all_embs = [], []
    n_ignored = 0

    for i, fpath in enumerate(files):
        with h5py.File(fpath, "r") as f:
            for sid in f.keys():
                sid_clean = str(int(float(sid)))
                if sid_clean not in manifest:
                    continue
                if field not in f[sid] or text_key not in f[sid]:
                    continue
                embs = f[sid][field][:].astype(np.float32)
                texts = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                         for x in f[sid][text_key][:]]
                for t, e in zip(texts, embs):
                    t = t.strip()
                    if not t:
                        continue
                    if ignore_patterns and any(p.search(t) for p in ignore_patterns):
                        n_ignored += 1
                        continue
                    all_texts.append(t)
                    all_embs.append(e)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} chunks, {len(all_texts):,} lines", flush=True)

    print(f"Loaded {len(all_texts):,} lines ({n_ignored:,} ignored) from {len(files)} chunks", flush=True)
    return all_texts, np.stack(all_embs), n_ignored


def load_ignore_patterns(path):
    patterns = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, re.IGNORECASE))
    return patterns


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def dedup(texts, embs):
    """Exact-match dedup. Returns unique texts, mean embeddings, counts."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, t in enumerate(texts):
        groups[t].append(i)

    unique_texts = sorted(groups.keys())
    unique_embs = np.stack([embs[groups[t]].mean(axis=0) for t in unique_texts])
    counts = np.array([len(groups[t]) for t in unique_texts], dtype=np.int32)
    print(f"Unique lines after dedup: {len(unique_texts):,}", flush=True)
    return unique_texts, unique_embs, counts


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

def cluster(embs, umap_dim, min_cluster_size):
    import umap
    import hdbscan

    print(f"UMAP: {embs.shape[1]}d -> {umap_dim}d ...", flush=True)
    reducer = umap.UMAP(
        n_components=umap_dim, metric="cosine",
        n_neighbors=30, min_dist=0.0, random_state=42, verbose=True,
    )
    reduced = reducer.fit_transform(embs)

    print(f"HDBSCAN: min_cluster_size={min_cluster_size} ...", flush=True)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean",
        cluster_selection_method="eom",
    )
    clusterer.fit(reduced)
    labels = clusterer.labels_

    n_clusters = labels.max() + 1
    n_noise = (labels == -1).sum()
    print(f"Clusters: {n_clusters:,}", flush=True)
    print(f"Noise points: {n_noise:,} ({100*n_noise/len(labels):.1f}%)", flush=True)
    return labels, reduced


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------

def build_codebook(texts, embs, counts, labels):
    cluster_ids = sorted(set(labels) - {-1})
    centroids, rep_texts, sizes = [], [], []

    for cid in cluster_ids:
        mask = labels == cid
        cluster_embs = embs[mask]
        cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]

        centroid = cluster_embs.mean(axis=0)
        centroids.append(centroid)
        sizes.append(int(mask.sum()))

        dists = 1 - (cluster_embs @ centroid) / (
            np.linalg.norm(cluster_embs, axis=1) * np.linalg.norm(centroid) + 1e-8)
        rep_texts.append(cluster_texts[dists.argmin()])

    return (np.array(cluster_ids, dtype=np.int32),
            np.stack(centroids).astype(np.float32),
            np.array(rep_texts, dtype=object),
            np.array(sizes, dtype=np.int32))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_umap_plot(embs_2d, labels, texts, counts, output_path, n_annotate_clusters=10, n_annotate_per=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 12))

    # Noise points
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(embs_2d[noise_mask, 0], embs_2d[noise_mask, 1],
                   c="lightgray", s=1, alpha=0.3, label="noise", rasterized=True)

    # Cluster points
    cluster_mask = ~noise_mask
    if cluster_mask.any():
        sc = ax.scatter(embs_2d[cluster_mask, 0], embs_2d[cluster_mask, 1],
                        c=labels[cluster_mask], cmap="tab20", s=2, alpha=0.5, rasterized=True)

    # Annotate top clusters
    cluster_ids = sorted(set(labels) - {-1})
    sizes = [(cid, (labels == cid).sum()) for cid in cluster_ids]
    sizes.sort(key=lambda x: -x[1])

    for cid, _ in sizes[:n_annotate_clusters]:
        mask = labels == cid
        cluster_embs = embs_2d[mask]
        cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]

        centroid = cluster_embs.mean(axis=0)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        nearest = dists.argsort()[:n_annotate_per]

        for idx in nearest:
            x, y = cluster_embs[idx]
            txt = cluster_texts[idx]
            if len(txt) > 60:
                txt = txt[:57] + "..."
            ax.annotate(txt, (x, y), fontsize=5, alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

    n_clusters = len(cluster_ids)
    n_noise_count = int(noise_mask.sum())
    ax.set_title(f"Line Embeddings — {n_clusters:,} clusters, {n_noise_count:,} noise")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UMAP plot: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Inspection + summary
# ---------------------------------------------------------------------------

def write_inspection(path, texts, embs, counts, labels):
    cluster_ids = sorted(set(labels) - {-1})
    with open(path, "w") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Cluster Inspection — {len(cluster_ids)} clusters\n")
        f.write(f"{'='*80}\n\n")

        for cid in cluster_ids:
            mask = labels == cid
            cluster_embs = embs[mask]
            cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
            cluster_counts = counts[mask]

            centroid = cluster_embs.mean(axis=0)
            dists = 1 - (cluster_embs @ centroid) / (
                np.linalg.norm(cluster_embs, axis=1) * np.linalg.norm(centroid) + 1e-8)
            order = dists.argsort()

            total_occ = int(cluster_counts.sum())
            f.write(f"--- Cluster {cid} | members: {mask.sum()} | "
                    f"total occurrences: {total_occ:,} ---\n")
            f.write("  Central:\n")
            for idx in order[:5]:
                f.write(f"    [{cluster_counts[idx]:>6}x] {cluster_texts[idx]}\n")
            if len(order) > 5:
                f.write("  Edge:\n")
                for idx in order[-3:]:
                    f.write(f"    [{cluster_counts[idx]:>6}x] {cluster_texts[idx]}\n")
            f.write("\n")


def write_summary(path, n_total, n_ignored, n_unique, n_clusters, n_noise, sizes):
    with open(path, "w") as f:
        f.write(f"Total lines:        {n_total:,}\n")
        f.write(f"Lines ignored:      {n_ignored:,}\n")
        f.write(f"Lines kept:         {n_total - n_ignored:,}\n")
        f.write(f"Unique after dedup: {n_unique:,}\n")
        f.write(f"Clusters:           {n_clusters:,}\n")
        f.write(f"Noise points:       {n_noise:,} ({100*n_noise/n_unique:.1f}%)\n")
        f.write(f"\nCluster size distribution:\n")
        f.write(f"  min:    {sizes.min()}\n")
        f.write(f"  median: {int(np.median(sizes))}\n")
        f.write(f"  mean:   {sizes.mean():.1f}\n")
        f.write(f"  max:    {sizes.max()}\n")
        f.write(f"  p90:    {int(np.percentile(sizes, 90))}\n")
        f.write(f"  p99:    {int(np.percentile(sizes, 99))}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    # Input (one of two modes)
    p.add_argument("--npz", default=None, help="encode.py output npz")
    p.add_argument("--h5_path", default=None, help="preprocess H5 for manifest filtering with --npz")
    p.add_argument("--h5_dir", default=None, help="Line_Embeddings_v2 dir (legacy mode)")
    p.add_argument("--manifest", default=None, help="study manifest (legacy mode)")
    p.add_argument("--field", default="study_findings", help="text field (legacy mode)")

    # Filtering
    p.add_argument("--ignore_file", default=None)

    # Clusterinig
    p.add_argument("--method", default="hdbscan", choices=["hdbscan", "kmeans"])
    p.add_argument("--k", type=int, default=10000, help="Number of clusters for kmeans")
    p.add_argument("--spherical", action="store_true", help="L2-normalize before kmeans")
    p.add_argument("--min_cluster_size", type=int, default=10)
    p.add_argument("--umap_dim", type=int, default=30)
    p.add_argument("--max_lines", type=int, default=None, help="Subsample unique lines before clustering")
    p.add_argument("--max_dist", type=float, default=None, help="Max cosine distance to centroid (spherical)")
    p.add_argument("--min_members", type=int, default=2, help="Dissolve clusters smaller than this")
    
    
    # Output
    p.add_argument("--output_dir", required=True)

    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ignore_patterns = []
    if args.ignore_file and Path(args.ignore_file).exists():
        ignore_patterns = load_ignore_patterns(args.ignore_file)
        print(f"Loaded {len(ignore_patterns)} ignore patterns")

    # Load
    n_ignored = 0
    if args.npz:
        texts, embs = load_from_npz(args.npz)
        if args.h5_path and args.manifest:
            manifest = set(str(int(float(x))) for x in
                           Path(args.manifest).read_text().strip().splitlines())
            allowed = set()
            with h5py.File(args.h5_path, "r") as f:
                for sid_raw in f.keys():
                    if str(int(float(sid_raw))) in manifest:
                        lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                                 for x in f[sid_raw][()]]
                        allowed.update(lines)
            print(f"Manifest filter: {len(manifest):,} studies, {len(allowed):,} unique lines", flush=True)
            keep = [t in allowed for t in texts]
            texts = [t for t, k in zip(texts, keep) if k]
            embs = embs[keep]
            print(f"After manifest filter: {len(texts):,}", flush=True)
        if ignore_patterns:
            keep = [not any(p.search(t) for p in ignore_patterns) for t in texts]
            n_ignored = sum(1 for k in keep if not k)
            texts = [t for t, k in zip(texts, keep) if k]
            embs = embs[keep]
            print(f"After ignore filter: {len(texts):,} ({n_ignored:,} ignored)", flush=True)
    elif args.h5_dir:
        if not args.manifest:
            raise ValueError("--manifest required with --h5_dir")
        texts, embs, n_ignored = load_from_h5(args.h5_dir, args.manifest, args.field, ignore_patterns)
    else:
        raise ValueError("Provide --npz or --h5_dir")

    n_total = len(texts) + n_ignored
    texts, embs, counts = dedup(texts, embs)
    n_unique = len(texts)

    if args.max_lines and n_unique > args.max_lines:
        idx = np.random.default_rng(42).choice(n_unique, args.max_lines, replace=False)
        idx.sort()
        texts = [texts[i] for i in idx]
        embs = embs[idx]
        counts = counts[idx]
        print(f"Subsampled to {args.max_lines:,} lines", flush=True)
    
    # Cluster
    if args.method == "hdbscan":
        labels, reduced = cluster(embs, args.umap_dim, args.min_cluster_size)
    else:
        from sklearn.cluster import KMeans
        embs_km = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8) if args.spherical else embs
        print(f"KMeans{'(spherical)' if args.spherical else ''}: k={args.k} ...", flush=True)
        km = KMeans(n_clusters=args.k, random_state=42, n_init=3, verbose=1)
        labels = km.fit_predict(embs_km)
        print(f"KMeans done. Inertia: {km.inertia_:.0f}", flush=True)

        if args.max_dist is not None:
            centroids_km = km.cluster_centers_
            for i in range(len(labels)):
                dist = 1 - np.dot(embs_km[i], centroids_km[labels[i]])
                if dist > args.max_dist:
                    labels[i] = -1
            n_filtered = (labels == -1).sum()
            print(f"Filtered {n_filtered:,} points beyond max_dist={args.max_dist}", flush=True)

        if args.min_members > 1:
            from collections import Counter
            counts = Counter(labels)
            small = {c for c, n in counts.items() if c != -1 and n < args.min_members}
            for i in range(len(labels)):
                if labels[i] in small:
                    labels[i] = -1
            labels = np.array(labels, dtype=np.int64)
            print(f"Dissolved {len(small):,} clusters below min_members={args.min_members}", flush=True)

    n_clusters = labels.max() + 1
    n_noise = int((labels == -1).sum())

    # 2D UMAP for visualization
    import umap
    print("UMAP: 2D for visualization ...", flush=True)
    vis_reducer = umap.UMAP(
        n_components=2, metric="cosine",
        n_neighbors=30, min_dist=0.1, random_state=42,
    )
    embs_2d = vis_reducer.fit_transform(embs)

    make_umap_plot(embs_2d, labels, texts, counts, out / "umap_clusters.png")

    # Codebook
    cids, centroids, rep_texts, sizes = build_codebook(texts, embs, counts, labels)
    np.savez(out / "codebook.npz", cluster_ids=cids, centroids=centroids,
             labels=rep_texts, sizes=sizes)
    print(f"Saved codebook: {len(cids)} clusters -> {out / 'codebook.npz'}")

    # Inspection + summary
    write_inspection(out / "inspect_clusters.txt", texts, embs, counts, labels)
    write_summary(out / "summary.txt", n_total, n_ignored, n_unique, n_clusters, n_noise, sizes)

    # Full mapping
    np.savez(out / "all_lines.npz", texts=np.array(texts, dtype=object),
             counts=counts, labels=labels, umap_2d=embs_2d)
    print(f"Saved all outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
