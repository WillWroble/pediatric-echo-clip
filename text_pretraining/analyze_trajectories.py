"""Analyze trajectory directions by diagnosis group.

For each embedding set:
1. Build consecutive pairs from val set
2. Compute delta directions
3. Group by cardiac_history
4. Compare within-group vs between-group cosine similarity

Usage:
    python -u analyze_trajectories.py \
        --h5_dir /lab-share/.../Line_Embeddings \
        --csv /lab-share/.../echo_reports_v2.csv \
        --embedding_dirs results/vicreg_50_50/eval results/traj_v1/eval results/joint_v1/eval \
        --labels vicreg traj_only joint \
        --output_dir results/trajectory_analysis
"""

import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def parse_study_date(s):
    try:
        return datetime.strptime(s, "%m%d%y%H%M%S")
    except Exception:
        return None


def build_pairs_from_embeddings(study_ids, Z, h5_dir):
    """Build consecutive pairs with their embeddings."""
    import h5py

    sid_set = set(study_ids)
    sid_to_idx = {s: i for i, s in enumerate(study_ids)}
    mrn_map, date_map = {}, {}

    for fpath in sorted(Path(h5_dir).glob("chunk_*.h5")):
        with h5py.File(fpath, "r") as f:
            for sid in f.keys():
                if sid in sid_set:
                    mrn_map[sid] = f[sid].attrs.get("mrn", "")
                    date_map[sid] = f[sid].attrs.get("study_date", "")

    patient_studies = defaultdict(list)
    for sid in study_ids:
        mrn = mrn_map.get(sid, "")
        dt = parse_study_date(date_map.get(sid, ""))
        if mrn and dt:
            patient_studies[mrn].append((dt, sid))

    pairs = []
    for mrn, studies in patient_studies.items():
        studies.sort()
        for i in range(len(studies) - 1):
            sid_t = studies[i][1]
            sid_t1 = studies[i + 1][1]
            if sid_t in sid_to_idx and sid_t1 in sid_to_idx:
                idx_t = sid_to_idx[sid_t]
                idx_t1 = sid_to_idx[sid_t1]
                delta = Z[idx_t1] - Z[idx_t]
                pairs.append({
                    "sid_t": sid_t,
                    "sid_t1": sid_t1,
                    "mrn": mrn,
                    "delta": delta,
                })
    return pairs


# ---------------------------------------------------------------------------
# Diagnosis grouping
# ---------------------------------------------------------------------------

def normalize_diagnosis(text):
    """Simple normalization: lowercase, strip whitespace."""
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip().lower()


def get_top_diagnoses(pairs, diag_map, min_count=50, max_groups=20):
    """Find most common diagnoses among trajectory pairs."""
    counts = defaultdict(int)
    for p in pairs:
        diag = diag_map.get(p["sid_t"])
        if diag:
            counts[diag] += 1

    ranked = sorted(counts.items(), key=lambda x: -x[1])
    top = [d for d, c in ranked if c >= min_count][:max_groups]
    return top


# ---------------------------------------------------------------------------
# Cosine similarity analysis
# ---------------------------------------------------------------------------

def compute_group_stats(pairs, diag_map, top_diagnoses):
    """Compute within-group and between-group cosine similarities."""
    # Group deltas by diagnosis
    diag_set = set(top_diagnoses)
    grouped = defaultdict(list)
    for p in pairs:
        diag = diag_map.get(p["sid_t"])
        if diag in diag_set:
            grouped[diag].append(p["delta"])

    # Convert to tensors
    grouped_t = {d: torch.from_numpy(np.stack(vs)).float()
                 for d, vs in grouped.items() if len(vs) >= 2}

    # Within-group: mean pairwise cosine between deltas in same diagnosis
    within = {}
    for diag, deltas in grouped_t.items():
        # Normalize
        normed = F.normalize(deltas, dim=1)
        # Pairwise cosine = dot product of normalized vectors
        sim_matrix = normed @ normed.T
        n = sim_matrix.shape[0]
        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        within[diag] = sim_matrix[mask].numpy()

    # Between-group: cosine between deltas from different diagnoses
    all_deltas = []
    all_labels = []
    for diag, deltas in grouped_t.items():
        all_deltas.append(deltas)
        all_labels.extend([diag] * len(deltas))
    all_deltas = torch.cat(all_deltas)
    all_labels = np.array(all_labels)
    all_normed = F.normalize(all_deltas, dim=1)

    # Sample between-group pairs (too many to compute all)
    rng = np.random.RandomState(42)
    n_between = 100_000
    idx_a = rng.randint(0, len(all_labels), n_between)
    idx_b = rng.randint(0, len(all_labels), n_between)
    diff_mask = all_labels[idx_a] != all_labels[idx_b]
    idx_a, idx_b = idx_a[diff_mask], idx_b[diff_mask]
    between_sims = (all_normed[idx_a] * all_normed[idx_b]).sum(dim=1).numpy()

    # Mean delta direction per group (the "king-queen" vector)
    mean_dirs = {}
    for diag, deltas in grouped_t.items():
        mean_dirs[diag] = F.normalize(deltas.mean(dim=0, keepdim=True), dim=1).squeeze()

    return within, between_sims, mean_dirs, {d: len(v) for d, v in grouped_t.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_within_vs_between(all_results, output_path):
    """Box plot of within-group vs between-group cosine similarity per model."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 6))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (label, within, between, counts) in zip(axes, all_results):
        # Aggregate within-group
        within_all = np.concatenate(list(within.values()))
        data = [within_all, between]
        bp = ax.boxplot(data, labels=["Within-dx", "Between-dx"],
                        showfliers=False, patch_artist=True)
        bp["boxes"][0].set_facecolor("tab:blue")
        bp["boxes"][1].set_facecolor("tab:orange")

        within_mean = np.mean(within_all)
        between_mean = np.mean(between)
        ax.set_title(f"{label}\nwithin={within_mean:.3f}  between={between_mean:.3f}")
        ax.set_ylabel("Cosine similarity")
        ax.axhline(0, color="gray", ls="--", lw=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_per_diagnosis(all_results, output_path):
    """Per-diagnosis within-group cosine similarity across models."""
    # Collect all diagnoses across models
    all_diags = set()
    for _, within, _, counts in all_results:
        all_diags.update(within.keys())

    # Sort by count in first model
    first_counts = all_results[0][3]
    diags = sorted(all_diags, key=lambda d: first_counts.get(d, 0), reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(diags))
    width = 0.8 / len(all_results)

    for i, (label, within, _, counts) in enumerate(all_results):
        means = [np.mean(within[d]) if d in within else 0 for d in diags]
        ax.bar(x + i * width, means, width, label=label, alpha=0.8)

    # Truncate long diagnosis strings
    short_diags = [d[:40] + "..." if len(d) > 40 else d for d in diags]
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(short_diags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean within-group cosine similarity")
    ax.set_title("Trajectory Direction Consistency by Diagnosis")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_mean_direction_similarity(all_results, output_path):
    """Heatmap of cosine similarity between mean trajectory directions per diagnosis."""
    for label, _, mean_dirs, counts in all_results:
        diags = sorted(mean_dirs.keys(), key=lambda d: counts.get(d, 0), reverse=True)[:15]
        vecs = torch.stack([mean_dirs[d] for d in diags])
        sim = (F.normalize(vecs, dim=1) @ F.normalize(vecs, dim=1).T).numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
        short = [d[:30] + "..." if len(d) > 30 else d for d in diags]
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Mean Trajectory Direction Similarity ({label})")
        plt.tight_layout()

        path = output_path.parent / f"{output_path.stem}_{label}{output_path.suffix}"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--csv", required=True, help="echo_reports_v2.csv")
    p.add_argument("--embedding_dirs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--min_count", type=int, default=50)
    p.add_argument("--max_groups", type=int, default=20)
    args = p.parse_args()

    assert len(args.embedding_dirs) == len(args.labels)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load diagnosis info
    print("Loading CSV...", flush=True)
    df = pd.read_csv(args.csv, dtype=str)
    df = df[["study_id", "cardiac_history"]].drop_duplicates(subset="study_id")
    diag_map = {}
    for _, row in df.iterrows():
        diag = normalize_diagnosis(row.get("cardiac_history", ""))
        if diag:
            diag_map[row["study_id"]] = diag
    print(f"  {len(diag_map):,} studies with diagnosis")

    all_results = []

    for emb_dir, label in zip(args.embedding_dirs, args.labels):
        print(f"\n--- {label} ---", flush=True)
        data = np.load(Path(emb_dir) / "embeddings.npz", allow_pickle=True)

        # Use val set
        Z = data["val"]
        study_ids = list(data["val_ids"])
        print(f"  {len(study_ids):,} val studies, {Z.shape[1]}d")

        # Build pairs
        print("  Building pairs...", flush=True)
        pairs = build_pairs_from_embeddings(study_ids, Z, args.h5_dir)
        print(f"  {len(pairs):,} trajectory pairs")

        # Get top diagnoses (use first model's ranking for consistency)
        if not all_results:
            top_diags = get_top_diagnoses(pairs, diag_map,
                                          min_count=args.min_count,
                                          max_groups=args.max_groups)
            print(f"  Top {len(top_diags)} diagnoses")

        within, between, mean_dirs, counts = compute_group_stats(pairs, diag_map, top_diags)

        within_all = np.concatenate(list(within.values())) if within else np.array([0])
        print(f"  Within-dx cosine: {np.mean(within_all):.4f} ± {np.std(within_all):.4f}")
        print(f"  Between-dx cosine: {np.mean(between):.4f} ± {np.std(between):.4f}")
        print(f"  Gap: {np.mean(within_all) - np.mean(between):.4f}")

        all_results.append((label, within, between, counts))

    # Plots
    plot_within_vs_between(all_results, out / "within_vs_between.png")
    plot_per_diagnosis(all_results, out / "per_diagnosis.png")
    plot_mean_direction_similarity(all_results, out / "direction_heatmap.png")

    # Summary
    summary = {}
    for label, within, between, counts in all_results:
        within_all = np.concatenate(list(within.values())) if within else np.array([0])
        summary[label] = {
            "within_mean": round(float(np.mean(within_all)), 4),
            "within_std": round(float(np.std(within_all)), 4),
            "between_mean": round(float(np.mean(between)), 4),
            "between_std": round(float(np.std(between)), 4),
            "gap": round(float(np.mean(within_all) - np.mean(between)), 4),
            "n_diagnoses": len(counts),
            "diagnosis_counts": {d: c for d, c in sorted(counts.items(), key=lambda x: -x[1])},
        }

    import json
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
