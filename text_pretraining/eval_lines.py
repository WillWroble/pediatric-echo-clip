"""Line-level UMAP and PC1 density for raw ClinicalBERT embeddings.

Visualizes the raw precomputed line embeddings (before any pretraining)
to expose temporal drift, demographic gradients, and distributional
structure at the line level.

Usage:
    python -u eval_lines.py \
        --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings \
        --output_dir results/raw_bert_lines \
        --n_sample 100000

    # With manifest + labels:
    python -u eval_lines.py \
        --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings \
        --manifest manifests/val_50_modern_nofetal.txt \
        --labels /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Labels_SG_Fyler_112025.csv \
        --death_mrn /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/death_mrn.csv \
        --output_dir results/raw_bert_lines \
        --n_sample 100000
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from umap import UMAP

from report_dataset import parse_study_date, TEXT_FIELDS


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def parse_age_years(age_str):
    if not age_str or not str(age_str).strip() or str(age_str).strip() == "nan":
        return np.nan
    s = str(age_str).strip().lower()
    try:
        if "year" in s:
            return float(s.split()[0])
        if s.endswith("m"):
            return float(s.rstrip("m")) / 12.0
        if s.endswith("d"):
            return float(s.rstrip("d")) / 365.25
        return float(s)
    except (ValueError, IndexError):
        return np.nan


def parse_float(val):
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


def to_fractional_year(sd_str):
    dt = parse_study_date(str(sd_str)) if sd_str else None
    if dt is None:
        return np.nan
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25


def load_lines_reservoir(h5_dir, manifest=None, n_sample=100000, seed=42):
    """Reservoir-sample n_sample line embeddings from HDF5 chunks."""
    keep = None
    if manifest:
        keep = set(Path(manifest).read_text().strip().splitlines())
        print(f"Manifest: {len(keep):,} studies")

    rng = np.random.RandomState(seed)
    reservoir_emb = None
    reservoir_meta = [None] * n_sample
    count = 0

    chunks = sorted(Path(h5_dir).glob("chunk_*.h5"))
    for i, fpath in enumerate(chunks):
        with h5py.File(fpath, "r") as f:
            for sid in f.keys():
                if keep is not None and sid not in keep:
                    continue
                grp = f[sid]
                arrays = [grp[field][:] for field in TEXT_FIELDS if field in grp]
                if not arrays:
                    continue
                lines = np.concatenate(arrays, axis=0).astype(np.float32)
                attrs = grp.attrs
                meta = {
                    "study_id": sid,
                    "mrn": str(attrs.get("mrn", "")),
                    "study_date": str(attrs.get("study_date", "")),
                    "age": str(attrs.get("age", "")),
                    "gender": str(attrs.get("gender", "")),
                    "weight_kg": str(attrs.get("weight_kg", "")),
                    "bsa": str(attrs.get("bsa", "")),
                    "n_lines": lines.shape[0],
                }
                if reservoir_emb is None:
                    dim = lines.shape[1]
                    reservoir_emb = np.empty((n_sample, dim), dtype=np.float32)

                for row in lines:
                    if count < n_sample:
                        reservoir_emb[count] = row
                        reservoir_meta[count] = meta
                    else:
                        j = rng.randint(0, count + 1)
                        if j < n_sample:
                            reservoir_emb[j] = row
                            reservoir_meta[j] = meta
                    count += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(chunks)} chunks, {count:,} lines seen", flush=True)

    n = min(count, n_sample)
    print(f"Total: {count:,} lines, sampled {n:,}")
    return reservoir_emb[:n], reservoir_meta[:n], count


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_colored(coords, values, title, path, cmap="viridis",
                 vmin=None, vmax=None, categorical=False, cat_labels=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    valid = ~np.isnan(values)
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1], s=0.3, alpha=0.1, c="lightgray")
    if categorical:
        for v in np.unique(values[valid]):
            m = valid & (values == v)
            label = cat_labels[int(v)] if cat_labels else str(v)
            ax.scatter(coords[m, 0], coords[m, 1], s=0.5, alpha=0.3, label=label)
        ax.legend(markerscale=10, fontsize=9)
    else:
        vmin = vmin if vmin is not None else np.nanpercentile(values, 2)
        vmax = vmax if vmax is not None else np.nanpercentile(values, 98)
        sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=0.5, alpha=0.3,
                        c=values[valid], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_pc1_density(pc1, years, path):
    """KDE of PC1 stratified by era (pre/post 2015)."""
    valid = ~np.isnan(years)
    pc1_v, years_v = pc1[valid], years[valid]

    pre = pc1_v[years_v < 2015]
    post = pc1_v[years_v >= 2015]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(pc1.min(), pc1.max(), 500)

    if len(pre) > 10:
        kde_pre = gaussian_kde(pre)
        ax.fill_between(x, kde_pre(x), alpha=0.4, color="tab:blue", label=f"Pre-2015 (n={len(pre):,})")
    if len(post) > 10:
        kde_post = gaussian_kde(post)
        ax.fill_between(x, kde_post(x), alpha=0.4, color="tab:orange", label=f"Post-2015 (n={len(post):,})")

    # Overall
    kde_all = gaussian_kde(pc1_v)
    ax.plot(x, kde_all(x), color="black", lw=1.5, ls="--", label=f"All (n={len(pc1_v):,})")

    ax.set_xlabel("PC1")
    ax.set_ylabel("Density")
    ax.set_title("Raw ClinicalBERT PC1 Density — Era Stratified")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_pc1_vs_year(pc1, years, path):
    """Scatter of PC1 vs study date."""
    valid = ~np.isnan(years)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(years[valid], pc1[valid], s=0.3, alpha=0.1, c="tab:blue")
    ax.set_xlabel("Study Date (year)")
    ax.set_ylabel("PC1")
    ax.set_title("Raw ClinicalBERT PC1 vs Study Date")
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_submap(coords, mask, color_vals, title, path, cmap="viridis",
                vmin=None, vmax=None):
    """UMAP of positive-only lines colored by a continuous variable."""
    sub_coords = coords[mask]
    sub_vals = color_vals[mask]
    valid = ~np.isnan(sub_vals)
    if valid.sum() < 10:
        print(f"  Skipping submap {title}: only {valid.sum()} valid points")
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    vmin = vmin if vmin is not None else np.nanpercentile(sub_vals, 2)
    vmax = vmax if vmax is not None else np.nanpercentile(sub_vals, 98)
    sc = ax.scatter(sub_coords[valid, 0], sub_coords[valid, 1], s=1.5, alpha=0.5,
                    c=sub_vals[valid], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Study Date")
    ax.set_title(f"{title} (n={int(mask.sum()):,})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--labels", default=None, help="Echo_Labels_SG_Fyler CSV")
    p.add_argument("--death_mrn", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_sample", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    print("Loading line embeddings (reservoir sampling)...", flush=True)
    embs, meta, total = load_lines_reservoir(
        args.h5_dir, args.manifest, args.n_sample, args.seed,
    )
    N, D = embs.shape
    print(f"Sampled {N:,} lines ({D}d) from {total:,} total", flush=True)

    # ---- metadata arrays ----
    years = np.array([to_fractional_year(m["study_date"]) for m in meta])
    ages = np.array([parse_age_years(m["age"]) for m in meta])
    n_lines = np.array([float(m["n_lines"]) for m in meta])

    gender = np.full(N, np.nan)
    for i, m in enumerate(meta):
        g = m["gender"].strip().lower()
        if g == "male":
            gender[i] = 1.0
        elif g == "female":
            gender[i] = 0.0

    weight = np.array([parse_float(m["weight_kg"]) for m in meta])
    bsa = np.array([parse_float(m["bsa"]) for m in meta])

    # ---- PCA ----
    print("\nFitting PCA...", flush=True)
    pca = PCA(n_components=10, random_state=args.seed)
    pcs = pca.fit_transform(embs)
    print(f"  Explained variance (top 10): {pca.explained_variance_ratio_.round(4)}")

    plot_pc1_density(pcs[:, 0], years, out / "pc1_density.png")
    plot_pc1_vs_year(pcs[:, 0], years, out / "pc1_vs_year.png")

    # ---- UMAP ----
    print("\nFitting UMAP...", flush=True)
    coords = UMAP(
        n_neighbors=30, min_dist=0.3, metric="cosine", random_state=args.seed,
    ).fit_transform(embs)

    plot_colored(coords, years, "Study Date", out / "umap_study_date.png", cmap="viridis")
    plot_colored(coords, ages, "Age (years)", out / "umap_age.png", cmap="plasma", vmin=0, vmax=25)
    plot_colored(coords, gender, "Gender", out / "umap_gender.png",
                 cmap="coolwarm", categorical=True, cat_labels={0: "Female", 1: "Male"})
    plot_colored(coords, weight, "Weight (kg)", out / "umap_weight.png", cmap="viridis")
    plot_colored(coords, bsa, "BSA (m²)", out / "umap_bsa.png", cmap="viridis")
    plot_colored(coords, n_lines, "Report Lines", out / "umap_nlines.png", cmap="inferno")

    # ---- diagnostic labels ----
    if args.labels:
        print("\nLoading diagnostic labels...", flush=True)
        label_df = pd.read_csv(args.labels, encoding="utf-8-sig")
        label_df["eid"] = label_df["eid"].astype(str)
        label_df["pid"] = label_df["pid"].astype(str)

        diag_cols = [c for c in [
            "Composite_Critical", "Composite_NonCritical",
            "VSD", "ASD", "TOF", "AVCD", "Coarct", "HLHS", "PDA",
            "TOF_ADJ", "AVCD_ADJ", "VSD_ADJ", "Coarct_ADJ",
        ] if c in label_df.columns]

        patient_labels = label_df.groupby("pid")[diag_cols].max()

        # Build mrn → pid bridge from label CSV eids that appear in h5
        mrn_to_pid = {}
        eid_to_pid = dict(zip(label_df["eid"], label_df["pid"]))
        # We need to scan h5 for the eid→mrn mapping
        print("  Building mrn→pid bridge...", flush=True)
        for fpath in sorted(Path(args.h5_dir).glob("chunk_*.h5")):
            with h5py.File(fpath, "r") as f:
                for sid in f.keys():
                    if sid in eid_to_pid:
                        mrn = str(f[sid].attrs.get("mrn", ""))
                        if mrn:
                            mrn_to_pid[mrn] = eid_to_pid[sid]
        print(f"  mrn→pid bridge: {len(mrn_to_pid):,} MRNs")

        val_mrns = [m["mrn"] for m in meta]
        for col in diag_cols:
            vals = np.array([
                float(patient_labels.loc[mrn_to_pid[mrn], col])
                if mrn in mrn_to_pid and mrn_to_pid[mrn] in patient_labels.index
                   and pd.notna(patient_labels.loc[mrn_to_pid[mrn], col])
                else np.nan
                for mrn in val_mrns
            ])
            n_pos = int(np.nansum(vals == 1))
            if n_pos < 5:
                print(f"  Skipping {col}: only {n_pos} positive lines")
                continue
            plot_colored(coords, vals, f"{col} (n_pos={n_pos})",
                         out / f"umap_dx_{col.lower()}.png", cmap="coolwarm",
                         categorical=True, cat_labels={0: "Negative", 1: "Positive"})

            # Sub-map: positive-only lines colored by study date
            pos_mask = vals == 1
            plot_submap(coords, pos_mask, years,
                        f"{col} — Positive Only × Study Date",
                        out / f"umap_sub_{col.lower()}.png")

    # ---- death ----
    if args.death_mrn:
        print("\nLoading death MRN data...", flush=True)
        death_df = pd.read_csv(args.death_mrn)
        death_df["mrn"] = death_df["mrn"].astype(str)
        dead_mrns = set(death_df["mrn"].str.lstrip("0"))

        death_vals = np.array([
            1.0 if m["mrn"].lstrip("0") in dead_mrns else 0.0
            for m in meta
        ])
        n_dead = int(death_vals.sum())
        print(f"  {n_dead:,} lines from deceased patients")

        fig, ax = plt.subplots(figsize=(8, 7))
        alive = death_vals == 0
        dead = death_vals == 1
        ax.scatter(coords[alive, 0], coords[alive, 1], s=0.3, alpha=0.08, c="lightgray")
        ax.scatter(coords[dead, 0], coords[dead, 1], s=8, alpha=0.7, c="red",
                   edgecolors="darkred", linewidths=0.2, zorder=5,
                   label=f"Deceased (n={n_dead:,})")
        ax.legend(markerscale=2, fontsize=9)
        ax.set_title(f"Deceased Patient Lines (n={n_dead:,})")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(out / "umap_death.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out / 'umap_death.png'}")

        # Sub-map: deceased-only lines colored by study date
        dead_mask = death_vals == 1
        plot_submap(coords, dead_mask, years,
                    "Deceased — Positive Only × Study Date",
                    out / "umap_sub_death.png")

    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
