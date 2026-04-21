"""Line-level UMAP and PCA for skip-gram or raw ClinicalBERT embeddings.

Two input modes:
  --npz: skip-gram embeddings (encode.py output), metadata from --h5_dir
  --h5_dir only: raw CLS from Line_Embeddings_v2

Usage (skip-gram):
    python -u eval_lines.py \
        --npz results/v3/line_embeddings.npz \
        --h5_dir /lab-share/.../Line_Embeddings_v2 \
        --manifest manifests/val_50_modern_nofetal.txt \
        --output_dir results/v3/eval_lines

Usage (legacy CLS):
    python -u eval_lines.py \
        --h5_dir /lab-share/.../Line_Embeddings_v2 \
        --manifest manifests/val_50_modern_nofetal.txt \
        --output_dir results/raw_bert_lines
"""

import argparse
import re
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from umap import UMAP


TEXT_FIELDS = [
    "summary", "study_findings", "history",
    "measurements", "cardiac_history", "reason_for_exam",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def merge_soft_wraps(lines):
    """Merge lines that are soft-wrapped (continuation of previous line)."""
    merged = []
    for line in lines:
        if merged and (line and line[0].islower() or (merged[-1] and merged[-1].endswith('-'))):
            merged[-1] = merged[-1].rstrip('-') + line
        else:
            merged.append(line)
    return merged
def parse_study_date(s):
    try:
        return datetime.strptime(s, "%m%d%y%H%M%S")
    except Exception:
        return None


def to_fractional_year(sd_str):
    dt = parse_study_date(str(sd_str)) if sd_str else None
    if dt is None:
        return np.nan
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25


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


def load_ignore_patterns(path):
    patterns = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, re.IGNORECASE))
    return patterns


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_npz_lines(npz_path, ignore_patterns):
    """Load unique lines + embeddings from encode.py output."""
    data = np.load(npz_path, allow_pickle=True)
    texts = data["lines"].astype(str).tolist()
    embs = data["embeddings"].astype(np.float32)
    if ignore_patterns:
        keep = [not any(p.search(t) for p in ignore_patterns) for t in texts]
        texts = [t for t, k in zip(texts, keep) if k]
        embs = embs[keep]
    print(f"Loaded {len(texts):,} lines from {npz_path}", flush=True)
    return texts, embs


def build_mean_metadata(h5_dir, target_lines, manifest, field, death_mrns):
    """Scan Line_Embeddings_v2 once, accumulate mean metadata per unique line."""
    target_set = set(target_lines)
    manifest_set = None
    if manifest:
        manifest_set = set(str(int(float(x)))
                           for x in Path(manifest).read_text().strip().splitlines())
        print(f"Manifest: {len(manifest_set):,} studies", flush=True)

    text_key = f"{field}_text"
    acc = {}
    for t in target_set:
        acc[t] = dict(sum_age=0.0, sum_year=0.0, n_male=0, n_female=0,
                       sum_weight=0.0, sum_bsa=0.0, sum_nlines=0.0,
                       n_death=0, count=0, n_age=0, n_year=0,
                       n_weight=0, n_bsa=0, mrns=set())

    chunks = sorted(Path(h5_dir).glob("chunk_*.h5"))
    for i, fpath in enumerate(chunks):
        with h5py.File(fpath, "r") as f:
            for sid in f.keys():
                if manifest_set is not None:
                    sid_clean = str(int(float(sid)))
                    if sid_clean not in manifest_set:
                        continue
                grp = f[sid]
                if field not in grp or text_key not in grp:
                    continue

                attrs = dict(grp.attrs)
                age = parse_age_years(attrs.get("age", ""))
                year = to_fractional_year(attrs.get("study_date", ""))
                gender = str(attrs.get("gender", "")).strip().lower()
                weight = parse_float(attrs.get("weight_kg", ""))
                bsa = parse_float(attrs.get("bsa", ""))
                mrn = str(attrs.get("mrn", ""))
                n_lines = sum(grp[fld].shape[0] for fld in TEXT_FIELDS if fld in grp)
                is_dead = mrn.lstrip("0") in death_mrns if death_mrns else False

                lines = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                         for x in grp[text_key][:]]

                lines = merge_soft_wraps(lines)  # add this

                for line in lines:
                    if line not in target_set:
                        continue
                    a = acc[line]
                    a["count"] += 1
                    if not np.isnan(age):
                        a["sum_age"] += age
                        a["n_age"] += 1
                    if not np.isnan(year):
                        a["sum_year"] += year
                        a["n_year"] += 1
                    if gender == "male":
                        a["n_male"] += 1
                    elif gender == "female":
                        a["n_female"] += 1
                    if not np.isnan(weight):
                        a["sum_weight"] += weight
                        a["n_weight"] += 1
                    if not np.isnan(bsa):
                        a["sum_bsa"] += bsa
                        a["n_bsa"] += 1
                    a["sum_nlines"] += n_lines
                    if is_dead:
                        a["n_death"] += 1
                    a["mrns"].add(mrn)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(chunks)} chunks", flush=True)

    meta = {}
    for t, a in acc.items():
        if a["count"] == 0:
            continue
        c = a["count"]
        meta[t] = {
            "mean_age": a["sum_age"] / a["n_age"] if a["n_age"] > 0 else np.nan,
            "mean_year": a["sum_year"] / a["n_year"] if a["n_year"] > 0 else np.nan,
            "prop_male": a["n_male"] / (a["n_male"] + a["n_female"])
                if (a["n_male"] + a["n_female"]) > 0 else np.nan,
            "mean_weight": a["sum_weight"] / a["n_weight"] if a["n_weight"] > 0 else np.nan,
            "mean_bsa": a["sum_bsa"] / a["n_bsa"] if a["n_bsa"] > 0 else np.nan,
            "mean_nlines": a["sum_nlines"] / c,
            "death_prop": a["n_death"] / c,
            "count": c,
            "mrns": a["mrns"],
        }
    print(f"Built metadata for {len(meta):,} / {len(target_set):,} lines", flush=True)
    return meta


def load_legacy_lines(h5_dir, manifest, field, ignore_patterns, death_mrns):
    """Load raw CLS embeddings + metadata from Line_Embeddings_v2.

    Deduplicates by line text, averages embeddings and metadata.
    """
    manifest_set = None
    if manifest:
        manifest_set = set(str(int(float(x)))
                           for x in Path(manifest).read_text().strip().splitlines())
        print(f"Manifest: {len(manifest_set):,} studies", flush=True)

    text_key = f"{field}_text"
    from collections import defaultdict
    acc = defaultdict(lambda: dict(
        sum_emb=None, sum_age=0.0, sum_year=0.0, n_male=0, n_female=0,
        sum_weight=0.0, sum_bsa=0.0, sum_nlines=0.0,
        n_death=0, count=0, n_age=0, n_year=0, n_weight=0, n_bsa=0, mrns=set(),
    ))

    chunks = sorted(Path(h5_dir).glob("chunk_*.h5"))
    for i, fpath in enumerate(chunks):
        with h5py.File(fpath, "r") as f:
            for sid in f.keys():
                if manifest_set is not None:
                    sid_clean = str(int(float(sid)))
                    if sid_clean not in manifest_set:
                        continue
                grp = f[sid]
                if field not in grp or text_key not in grp:
                    continue

                attrs = dict(grp.attrs)
                age = parse_age_years(attrs.get("age", ""))
                year = to_fractional_year(attrs.get("study_date", ""))
                gender = str(attrs.get("gender", "")).strip().lower()
                weight = parse_float(attrs.get("weight_kg", ""))
                bsa = parse_float(attrs.get("bsa", ""))
                mrn = str(attrs.get("mrn", ""))
                n_lines = sum(grp[fld].shape[0] for fld in TEXT_FIELDS if fld in grp)
                is_dead = mrn.lstrip("0") in death_mrns if death_mrns else False

                embs = grp[field][:].astype(np.float32)
                lines = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                         for x in grp[text_key][:]]

                lines = merge_soft_wraps(lines)  # add this

                for line, emb in zip(lines, embs):
                    if ignore_patterns and any(p.search(line) for p in ignore_patterns):
                        continue
                    a = acc[line]
                    if a["sum_emb"] is None:
                        a["sum_emb"] = emb.copy()
                    else:
                        a["sum_emb"] += emb
                    a["count"] += 1
                    if not np.isnan(age):
                        a["sum_age"] += age
                        a["n_age"] += 1
                    if not np.isnan(year):
                        a["sum_year"] += year
                        a["n_year"] += 1
                    if gender == "male":
                        a["n_male"] += 1
                    elif gender == "female":
                        a["n_female"] += 1
                    if not np.isnan(weight):
                        a["sum_weight"] += weight
                        a["n_weight"] += 1
                    if not np.isnan(bsa):
                        a["sum_bsa"] += bsa
                        a["n_bsa"] += 1
                    a["sum_nlines"] += n_lines
                    if is_dead:
                        a["n_death"] += 1
                    a["mrns"].add(mrn)

        if (i + 1) % 50 == 0:
            matched = sum(1 for a in acc.values() if a["count"] > 0)
            print(f"  {i+1}/{len(chunks)} chunks, {matched:,} unique lines", flush=True)

    texts, embs_list, meta = [], [], {}
    for line, a in acc.items():
        if a["count"] == 0:
            continue
        c = a["count"]
        texts.append(line)
        embs_list.append(a["sum_emb"] / c)
        meta[line] = {
            "mean_age": a["sum_age"] / a["n_age"] if a["n_age"] > 0 else np.nan,
            "mean_year": a["sum_year"] / a["n_year"] if a["n_year"] > 0 else np.nan,
            "prop_male": a["n_male"] / (a["n_male"] + a["n_female"])
                if (a["n_male"] + a["n_female"]) > 0 else np.nan,
            "mean_weight": a["sum_weight"] / a["n_weight"] if a["n_weight"] > 0 else np.nan,
            "mean_bsa": a["sum_bsa"] / a["n_bsa"] if a["n_bsa"] > 0 else np.nan,
            "mean_nlines": a["sum_nlines"] / c,
            "death_prop": a["n_death"] / c,
            "count": c,
            "mrns": a["mrns"],
        }

    embs = np.stack(embs_list)
    print(f"Loaded {len(texts):,} unique lines from {len(chunks)} chunks", flush=True)
    return texts, embs, meta


# ---------------------------------------------------------------------------
# Plotting (matching original eval_lines.py style)
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

    kde_all = gaussian_kde(pc1_v)
    ax.plot(x, kde_all(x), color="black", lw=1.5, ls="--", label=f"All (n={len(pc1_v):,})")

    ax.set_xlabel("PC1")
    ax.set_ylabel("Density")
    ax.set_title("PC1 Density — Era Stratified")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_pc1_vs_year(pc1, years, path):
    valid = ~np.isnan(years)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(years[valid], pc1[valid], s=0.3, alpha=0.1, c="tab:blue")
    ax.set_xlabel("Study Date (year)")
    ax.set_ylabel("PC1")
    ax.set_title("PC1 vs Study Date")
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_submap(coords, mask, color_vals, title, path, cmap="viridis",
                vmin=None, vmax=None):
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
    p.add_argument("--npz", default=None, help="encode.py output npz (skip-gram mode)")
    p.add_argument("--h5_dir", required=True, help="Line_Embeddings_v2 dir (metadata + legacy CLS)")
    p.add_argument("--manifest", default=None)
    p.add_argument("--field", default="study_findings")
    p.add_argument("--ignore_file", default=None)
    p.add_argument("--labels", default=None, help="Echo_Labels_SG_Fyler CSV")
    p.add_argument("--death_mrn", default=None)
    p.add_argument("--n_sample", type=int, default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ignore_patterns = []
    if args.ignore_file and Path(args.ignore_file).exists():
        ignore_patterns = load_ignore_patterns(args.ignore_file)
        print(f"Loaded {len(ignore_patterns)} ignore patterns")

    death_mrns = set()
    if args.death_mrn:
        death_df = pd.read_csv(args.death_mrn)
        death_df["mrn"] = death_df["mrn"].astype(str)
        death_mrns = set(death_df["mrn"].str.lstrip("0"))
        print(f"Loaded {len(death_mrns):,} death MRNs", flush=True)

    # ---- load embeddings + metadata ----
    if args.npz:
        texts, embs = load_npz_lines(args.npz, ignore_patterns)
        if args.n_sample and len(texts) > args.n_sample:
            idx = np.random.default_rng(args.seed).choice(len(texts), args.n_sample, replace=False)
            idx.sort()
            texts = [texts[i] for i in idx]
            embs = embs[idx]
            print(f"Subsampled to {len(texts):,} lines", flush=True)
        print("Building metadata from Line_Embeddings_v2...", flush=True)
        meta = build_mean_metadata(args.h5_dir, texts, args.manifest,
                                   args.field, death_mrns)

        # Filter to lines that appeared in manifest studies
        keep_idx = [i for i, t in enumerate(texts) if t in meta]
        texts = [texts[i] for i in keep_idx]
        embs = embs[keep_idx]
        print(f"After manifest filter: {len(texts):,} lines", flush=True)
    else:
        texts, embs, meta = load_legacy_lines(args.h5_dir, args.manifest,
                                               args.field, ignore_patterns,
                                               death_mrns)
        if args.n_sample and len(texts) > args.n_sample:
            idx = np.random.default_rng(args.seed).choice(len(texts), args.n_sample, replace=False)
            idx.sort()
            texts = [texts[i] for i in idx]
            embs = embs[idx]
            print(f"Subsampled to {len(texts):,} lines", flush=True)

    N = len(texts)
    print(f"\n{N:,} unique lines, {embs.shape[1]}d embeddings", flush=True)

    # ---- metadata arrays ----
    years = np.array([meta[t]["mean_year"] if t in meta else np.nan for t in texts])
    ages = np.array([meta[t]["mean_age"] if t in meta else np.nan for t in texts])
    prop_male = np.array([meta[t]["prop_male"] if t in meta else np.nan for t in texts])
    weights = np.array([meta[t]["mean_weight"] if t in meta else np.nan for t in texts])
    bsas = np.array([meta[t]["mean_bsa"] if t in meta else np.nan for t in texts])
    n_lines = np.array([meta[t]["mean_nlines"] if t in meta else np.nan for t in texts])
    death_prop = np.array([meta[t]["death_prop"] if t in meta else 0.0 for t in texts])

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
        n_neighbors=10, min_dist=0.1, metric="cosine", random_state=args.seed,
    ).fit_transform(embs)

    plot_colored(coords, years, "Mean Study Date", out / "umap_study_date.png", cmap="viridis")
    plot_colored(coords, ages, "Mean Age (years)", out / "umap_age.png", cmap="plasma", vmin=0, vmax=25)
    plot_colored(coords, prop_male, "Proportion Male", out / "umap_gender.png", cmap="coolwarm")
    plot_colored(coords, weights, "Mean Weight (kg)", out / "umap_weight.png", cmap="viridis")
    plot_colored(coords, bsas, "Mean BSA (m²)", out / "umap_bsa.png", cmap="viridis")
    plot_colored(coords, n_lines, "Mean Report Lines", out / "umap_nlines.png", cmap="inferno")

    # ---- death ----
    if death_mrns:
        print("\nPlotting death overlay...", flush=True)
        high_death = death_prop > 0.1
        low_death = ~high_death
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(coords[low_death, 0], coords[low_death, 1], s=0.3, alpha=0.08, c="lightgray")
        n_high = int(high_death.sum())
        ax.scatter(coords[high_death, 0], coords[high_death, 1], s=8, alpha=0.7, c="red",
                   edgecolors="darkred", linewidths=0.2, zorder=5,
                   label=f"Death prop >10% (n={n_high:,})")
        ax.legend(markerscale=2, fontsize=9)
        ax.set_title(f"Lines Associated with Deceased Patients (n={n_high:,})")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(out / "umap_death.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out / 'umap_death.png'}")

        plot_colored(coords, death_prop, "Death Proportion", out / "umap_death_prop.png",
                     cmap="Reds", vmin=0, vmax=0.5)

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

        eid_to_pid = dict(zip(label_df["eid"], label_df["pid"]))
        mrn_to_pid = {}
        print("  Building mrn→pid bridge...", flush=True)
        for fpath in sorted(Path(args.h5_dir).glob("chunk_*.h5")):
            with h5py.File(fpath, "r") as f:
                for sid in f.keys():
                    if sid in eid_to_pid:
                        mrn = str(f[sid].attrs.get("mrn", ""))
                        if mrn:
                            mrn_to_pid[mrn] = eid_to_pid[sid]
        print(f"  mrn→pid bridge: {len(mrn_to_pid):,} MRNs")

        for col in diag_cols:
            vals = np.full(N, np.nan)
            for i, t in enumerate(texts):
                if t not in meta:
                    continue
                mrns = meta[t]["mrns"]
                pos = any(
                    mrn in mrn_to_pid
                    and mrn_to_pid[mrn] in patient_labels.index
                    and patient_labels.loc[mrn_to_pid[mrn], col] == 1
                    for mrn in mrns
                )
                vals[i] = 1.0 if pos else 0.0

            n_pos = int(np.nansum(vals == 1))
            if n_pos < 5:
                print(f"  Skipping {col}: only {n_pos} positive lines")
                continue
            plot_colored(coords, vals, f"{col} (n_pos={n_pos})",
                         out / f"umap_dx_{col.lower()}.png", cmap="coolwarm",
                         categorical=True, cat_labels={0: "Negative", 1: "Positive"})

            pos_mask = vals == 1
            plot_submap(coords, pos_mask, years,
                        f"{col} — Positive Only × Study Date",
                        out / f"umap_sub_{col.lower()}.png")

    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
