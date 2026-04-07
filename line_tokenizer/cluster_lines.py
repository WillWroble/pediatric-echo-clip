"""Cluster ClinicalBERT line embeddings into a semantic codebook.

Loads line embeddings + text from Line_Embeddings_v2 H5 chunks,
filters by manifest and ignore patterns, deduplicates by exact text,
reduces with UMAP, clusters with HDBSCAN, and outputs a codebook.

Usage:
    python -u cluster_lines.py \
        --h5_dir /lab-share/.../Line_Embeddings_v2 \
        --manifest text_pretraining/manifests/train_50_modern_nofetal.txt \
        --field study_findings \
        --ignore_file line_tokenizer/ignore_patterns.txt \
        --min_cluster_size 10 \
        --umap_dim 30 \
        --output_dir line_tokenizer/results/findings_2020
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from collections import Counter


def load_ignore_patterns(path):
    """Load regex patterns from file, one per line. Blank/comment lines skipped."""
    patterns = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, re.IGNORECASE))
    return patterns


def should_ignore(text, patterns):
    return any(p.search(text) for p in patterns)


def load_lines(h5_dir, manifest_path, field, ignore_patterns):
    """Load all line embeddings + text for a field, filtered by manifest and ignore."""
    manifest = set(x.strip().strip("'\"") for x in Path(manifest_path).read_text().strip().splitlines())
    print(f"Manifest: {len(manifest):,} studies")

    files = sorted(Path(h5_dir).glob("chunk_*.h5"))
    text_key = f"{field}_text"

    all_texts = []
    all_embs = []
    n_ignored = 0
    n_total = 0
    n_studies = 0

    for i, fpath in enumerate(files):
        with h5py.File(fpath, "r") as f:
            for sid in f.keys():
                if sid not in manifest:
                    continue
                if field not in f[sid] or text_key not in f[sid]:
                    continue

                embs = f[sid][field][:].astype(np.float32)
                raw = f[sid][text_key][:]
                texts = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]
                n_studies += 1

                for t, e in zip(texts, embs):
                    t = t.strip()
                    n_total += 1
                    if not t:
                        continue
                    if ignore_patterns and should_ignore(t, ignore_patterns):
                        n_ignored += 1
                        continue
                    all_texts.append(t)
                    all_embs.append(e)

        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{len(files)} chunks, {len(all_texts):,} lines so far")

    print(f"Studies loaded: {n_studies:,}")
    print(f"Total lines: {n_total:,}")
    print(f"Lines ignored: {n_ignored:,}")
    print(f"Lines kept: {len(all_texts):,}")
    return all_texts, np.stack(all_embs)


def dedup(texts, embs):
    """Exact-match dedup. Returns unique texts, their embeddings, and counts."""
    counts = Counter(texts)
    seen = {}
    unique_texts = []
    unique_embs = []

    for t, e in zip(texts, embs):
        if t not in seen:
            seen[t] = len(unique_texts)
            unique_texts.append(t)
            unique_embs.append(e)

    unique_counts = np.array([counts[t] for t in unique_texts], dtype=np.int32)
    print(f"Unique lines after dedup: {len(unique_texts):,}")
    return unique_texts, np.stack(unique_embs), unique_counts


def cluster(embs, umap_dim, min_cluster_size):
    """UMAP reduce then HDBSCAN cluster. Returns labels array."""
    import umap
    import hdbscan

    print(f"UMAP: {embs.shape[1]}d -> {umap_dim}d ...", flush=True)
    reducer = umap.UMAP(
        n_components=umap_dim,
        metric="cosine",
        n_neighbors=30,
        min_dist=0.0,
        random_state=42,
        verbose=True,
    )
    reduced = reducer.fit_transform(embs)

    print(f"HDBSCAN: min_cluster_size={min_cluster_size} ...", flush=True)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",  # euclidean on UMAP output (already cosine-shaped)
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(reduced)
    labels = clusterer.labels_

    n_clusters = labels.max() + 1
    n_noise = (labels == -1).sum()
    print(f"Clusters: {n_clusters:,}")
    print(f"Noise points: {n_noise:,} ({100*n_noise/len(labels):.1f}%)")
    return labels, reduced


def build_codebook(texts, embs, counts, labels):
    """Build codebook: centroid + representative text per cluster."""
    cluster_ids = sorted(set(labels) - {-1})
    centroids = []
    rep_texts = []
    sizes = []

    for cid in cluster_ids:
        mask = labels == cid
        cluster_embs = embs[mask]
        cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        cluster_counts = counts[mask]

        centroid = cluster_embs.mean(axis=0)
        centroids.append(centroid)
        sizes.append(int(mask.sum()))

        # representative = closest to centroid
        dists = 1 - (cluster_embs @ centroid) / (
            np.linalg.norm(cluster_embs, axis=1) * np.linalg.norm(centroid) + 1e-8
        )
        rep_idx = dists.argmin()
        rep_texts.append(cluster_texts[rep_idx])

    return (
        np.array(cluster_ids, dtype=np.int32),
        np.stack(centroids).astype(np.float32),
        np.array(rep_texts, dtype=object),
        np.array(sizes, dtype=np.int32),
    )


def write_inspection(path, texts, embs, counts, labels):
    """Write human-readable cluster inspection file."""
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
                np.linalg.norm(cluster_embs, axis=1) * np.linalg.norm(centroid) + 1e-8
            )
            order = dists.argsort()

            total_occurrences = int(cluster_counts.sum())
            f.write(f"--- Cluster {cid} | members: {mask.sum()} | "
                    f"total occurrences: {total_occurrences:,} ---\n")

            f.write("  Central:\n")
            for idx in order[:5]:
                f.write(f"    [{cluster_counts[idx]:>6}x] {cluster_texts[idx]}\n")

            if len(order) > 5:
                f.write("  Edge:\n")
                for idx in order[-3:]:
                    f.write(f"    [{cluster_counts[idx]:>6}x] {cluster_texts[idx]}\n")

            f.write("\n")


def write_summary(path, n_total, n_ignored, n_kept, n_unique, n_clusters,
                  n_noise, sizes):
    with open(path, "w") as f:
        f.write(f"Total lines:           {n_total:,}\n")
        f.write(f"Lines ignored:         {n_ignored:,}\n")
        f.write(f"Lines kept:            {n_kept:,}\n")
        f.write(f"Unique after dedup:    {n_unique:,}\n")
        f.write(f"Clusters:              {n_clusters:,}\n")
        f.write(f"Noise points:          {n_noise:,} ({100*n_noise/n_unique:.1f}%)\n")
        f.write(f"\nCluster size distribution:\n")
        f.write(f"  min:    {sizes.min()}\n")
        f.write(f"  median: {int(np.median(sizes))}\n")
        f.write(f"  mean:   {sizes.mean():.1f}\n")
        f.write(f"  max:    {sizes.max()}\n")
        f.write(f"  p90:    {int(np.percentile(sizes, 90))}\n")
        f.write(f"  p99:    {int(np.percentile(sizes, 99))}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--field", default="study_findings")
    p.add_argument("--ignore_file", default=None)
    p.add_argument("--min_cluster_size", type=int, default=10)
    p.add_argument("--umap_dim", type=int, default=30)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ignore_patterns = []
    if args.ignore_file and Path(args.ignore_file).exists():
        ignore_patterns = load_ignore_patterns(args.ignore_file)
        print(f"Loaded {len(ignore_patterns)} ignore patterns")

    texts, embs = load_lines(args.h5_dir, args.manifest, args.field, ignore_patterns)
    n_total = len(texts)

    texts, embs, counts = dedup(texts, embs)
    n_unique = len(texts)

    labels, reduced = cluster(embs, args.umap_dim, args.min_cluster_size)

    n_clusters = labels.max() + 1
    n_noise = int((labels == -1).sum())

    # codebook
    cids, centroids, rep_texts, sizes = build_codebook(texts, embs, counts, labels)
    np.savez(
        out / "codebook.npz",
        cluster_ids=cids,
        centroids=centroids,
        labels=rep_texts,
        sizes=sizes,
    )
    print(f"Saved codebook: {len(cids)} clusters -> {out / 'codebook.npz'}")

    # inspection
    write_inspection(out / "inspect_clusters.txt", texts, embs, counts, labels)
    print(f"Saved {out / 'inspect_clusters.txt'}")

    # summary
    write_summary(out / "summary.txt", n_total, 0, n_total, n_unique,
                  n_clusters, n_noise, sizes)
    print(f"Saved {out / 'summary.txt'}")

    # also save the full mapping for analysis
    np.savez(
        out / "all_lines.npz",
        texts=np.array(texts, dtype=object),
        counts=counts,
        labels=labels,
    )
    print(f"Saved {out / 'all_lines.npz'}")


if __name__ == "__main__":
    main()
