"""Analyze embedding space quality from npz files."""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def sample_by_study(embs, study_ids, n_studies=200):
    unique = np.unique(study_ids)
    chosen = np.random.choice(unique, min(n_studies, len(unique)), replace=False)
    mask = np.isin(study_ids, chosen)
    return embs[mask], study_ids[mask]


def similarity_ratios(embs, study_ids):
    sim = cosine_similarity(embs)
    ids = np.array(study_ids)
    same = (ids[:, None] == ids[None, :])
    np.fill_diagonal(same, False)
    diff = ~(ids[:, None] == ids[None, :])
    intra = sim[same].mean()
    inter = sim[diff].mean()
    return intra, inter, intra / inter if inter != 0 else float("inf")


def knn_enrichment(embs, study_ids, k=10):
    sim = cosine_similarity(embs)
    np.fill_diagonal(sim, -1)
    ids = np.array(study_ids)
    enrichments = []
    for i in range(len(embs)):
        neighbors = np.argsort(sim[i])[-k:]
        enrichments.append((ids[neighbors] == ids[i]).sum() / k)
    return np.mean(enrichments)


def mean_first_neighbor_rank(embs, study_ids):
    sim = cosine_similarity(embs)
    np.fill_diagonal(sim, -1)
    ids = np.array(study_ids)
    ranks = []
    for i in range(len(embs)):
        sorted_idx = np.argsort(sim[i])[::-1]
        sorted_ids = ids[sorted_idx]
        same = np.where(sorted_ids == ids[i])[0]
        if len(same) > 0:
            ranks.append(same[0] + 1)
    return np.mean(ranks), np.median(ranks)


def within_study_variance(embs, study_ids):
    ids = np.array(study_ids)
    variances = []
    for sid in np.unique(ids):
        mask = ids == sid
        if mask.sum() < 2:
            continue
        variances.append(np.var(embs[mask], axis=0).mean())
    return np.mean(variances)


def plot_umap(embs, study_ids, title, ax):
    if not HAS_UMAP:
        ax.text(0.5, 0.5, "umap-learn not installed", ha="center", va="center")
        return
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embs)
    uid, int_ids = np.unique(study_ids, return_inverse=True)
    ax.scatter(coords[:, 0], coords[:, 1], c=int_ids % 20, cmap="tab20", s=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_similarity_histograms(embs, study_ids, title, ax):
    sim = cosine_similarity(embs)
    ids = np.array(study_ids)
    same = (ids[:, None] == ids[None, :])
    np.fill_diagonal(same, False)
    diff = ~same & ~np.eye(len(ids), dtype=bool)
    intra_sims = sim[same]
    inter_sims = np.random.choice(sim[diff], min(len(intra_sims) * 5, diff.sum()), replace=False)
    ax.hist(inter_sims, bins=80, alpha=0.5, label="inter-study", density=True)
    ax.hist(intra_sims, bins=80, alpha=0.5, label="intra-study", density=True)
    ax.set_title(title)
    ax.legend()


def main(args):
    data = np.load(args.input, allow_pickle=True)
    embs_all = data["embeddings"].astype(np.float32)
    sids_all = data["study_ids"]

    np.random.seed(42)
    embs, sids = sample_by_study(embs_all, sids_all, args.n_studies)
    print(f"Sampled {args.n_studies} studies: {len(embs)} videos, {embs.shape[1]}d")
    print(f"Avg videos/study: {len(embs) / args.n_studies:.0f}")

    intra, inter, ratio = similarity_ratios(embs, sids)
    knn = knn_enrichment(embs, sids, k=10)
    var = within_study_variance(embs, sids)
    rank_mean, rank_med = mean_first_neighbor_rank(embs, sids)

    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 47)
    print(f"{'Intra-study similarity':<35} {intra:>10.4f}")
    print(f"{'Inter-study similarity':<35} {inter:>10.4f}")
    print(f"{'Intra/Inter ratio':<35} {ratio:>10.2f}x")
    print(f"{'KNN-10 study enrichment':<35} {knn:>10.4f}")
    print(f"{'Within-study variance':<35} {var:>10.6f}")
    print(f"{'First same-study neighbor (mean)':<35} {rank_mean:>10.1f}")
    print(f"{'First same-study neighbor (median)':<35} {rank_med:>10.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_similarity_histograms(embs, sids, "Cosine similarity", axes[0])
    plot_umap(embs, sids, "UMAP", axes[1])

    plt.tight_layout()
    plt.savefig(args.output_fig, dpi=150)
    print(f"\nSaved figure to {args.output_fig}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help=".npz with embeddings + study_ids")
    p.add_argument("--output_fig", default="analysis.png")
    p.add_argument("--n_studies", type=int, default=200)
    main(p.parse_args())
