"""Label drift analysis: are diagnostic labels / mortality correlated with study era?

Usage:
    python -u eval_label_drift.py \
        --val_manifest manifests/val_50_modern_nofetal.txt \
        --h5_dir /lab-share/.../Line_Embeddings \
        --labels Echo_Labels_SG_Fyler_112025.csv \
        --death_mrn death_mrn.csv \
        --output_dir results/eval_label_drift
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report_dataset import load_meta_only, parse_study_date


YEAR_BINS = range(1975, 2026, 5)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histogram(years_pos, years_all, title, path, pos_label="Positive"):
    bins = list(YEAR_BINS)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(years_all, bins=bins, color="steelblue", alpha=0.5, label="All studies")
    ax.hist(years_pos, bins=bins, color="tomato",    alpha=0.7, label=pos_label)
    ax.set_xlabel("Year")
    ax.set_ylabel("Studies")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_heatmap(matrix, dx_labels, year_labels, path):
    fig, ax = plt.subplots(figsize=(14, max(6, len(dx_labels) * 0.28)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(year_labels)))
    ax.set_xticklabels(year_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(dx_labels)))
    ax.set_yticklabels(dx_labels, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Fraction positive studies")
    ax.set_title("Label prevalence by era (fraction of studies from positive patients)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--h5_dir",       required=True)
    p.add_argument("--labels",       default=None)
    p.add_argument("--death_mrn",    default=None)
    p.add_argument("--output_dir",   required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load study IDs from manifest
    val_ids = Path(args.val_manifest).read_text().strip().splitlines()

    # Load metadata only (fast — no line embeddings)
    data     = load_meta_only(args.h5_dir)
    val_ids  = [s for s in val_ids if s in data]
    print(f"Val studies: {len(val_ids):,}", flush=True)

    meta = []
    for sid in val_ids:
        mrn, sd_str = data[sid]
        dt = parse_study_date(sd_str)
        meta.append(dict(mrn=mrn, study_date=dt,
                         year=dt.year if dt else np.nan))

    all_years = np.array([m["year"] for m in meta])

    # Diagnostic labels
    if args.labels:
        print("Loading diagnostic labels...", flush=True)
        label_df = pd.read_csv(args.labels, encoding="utf-8-sig")
        label_df["eid"] = label_df["eid"].astype(str)
        label_df["pid"] = label_df["pid"].astype(str)

        diag_cols = [c for c in label_df.columns
                     if c not in {"eid", "pid", "Gender", "Age"}
                     and not c.strip().lstrip("-").isdigit()]

        patient_labels = label_df.groupby("pid")[diag_cols].max()
        eid_to_pid     = dict(zip(label_df["eid"], label_df["pid"]))

        def label_vector(ids):
            Y = np.full((len(ids), len(diag_cols)), np.nan, dtype=np.float32)
            for i, sid in enumerate(ids):
                pid = eid_to_pid.get(sid)
                if pid is None or pid not in patient_labels.index:
                    continue
                Y[i] = patient_labels.loc[pid, diag_cols].values.astype(np.float32)
            return Y

        Y = label_vector(val_ids)

        # Heatmap: fraction positive per (dx, year_bin)
        bins       = list(YEAR_BINS)
        bin_labels = [f"{b}–{b+4}" for b in bins]
        n_bins     = len(bins)
        matrix     = np.full((len(diag_cols), n_bins), np.nan)

        for j, col in enumerate(diag_cols):
            y     = Y[:, j]
            valid = ~np.isnan(y) & ~np.isnan(all_years)
            for b, lo in enumerate(bins):
                in_bin = valid & (all_years >= lo) & (all_years < lo + 5)
                if in_bin.sum() == 0:
                    continue
                matrix[j, b] = y[in_bin].mean()

        plot_heatmap(matrix, diag_cols, bin_labels, out / "heatmap_label_drift.png")

        # Save heatmap data as CSV
        df_heat = pd.DataFrame(matrix, index=diag_cols, columns=bin_labels)
        df_heat.to_csv(out / "heatmap_label_drift.csv")
        print(f"Saved {out / 'heatmap_label_drift.csv'}")

        # Per-dx histograms
        hist_dir = out / "histograms"
        hist_dir.mkdir(exist_ok=True)
        for j, col in enumerate(diag_cols):
            y     = Y[:, j]
            valid = ~np.isnan(y) & ~np.isnan(all_years)
            pos   = valid & (y == 1)
            n_pos = int(pos.sum())
            if n_pos < 2:
                continue
            plot_histogram(
                all_years[pos], all_years[valid],
                f"{col}  (n={n_pos} positive studies)",
                hist_dir / f"hist_{col.lower().replace(' ', '_').replace('+', '_')}.png",
                pos_label=f"{col} positive",
            )

    # Death
    if args.death_mrn:
        print("Loading death data...", flush=True)
        death_df  = pd.read_csv(args.death_mrn)
        death_df["mrn"] = death_df["mrn"].astype(str)
        dead_mrns = set(death_df["mrn"].str.lstrip("0"))

        patient_latest = {}
        for i, m in enumerate(meta):
            mrn_norm = m["mrn"].lstrip("0")
            if mrn_norm not in dead_mrns or m["study_date"] is None:
                continue
            if mrn_norm not in patient_latest or m["study_date"] > patient_latest[mrn_norm][0]:
                patient_latest[mrn_norm] = (m["study_date"], i)

        death_vals = np.zeros(len(val_ids), dtype=np.float32)
        for _, idx in patient_latest.values():
            death_vals[idx] = 1.0
        n_dead = len(patient_latest)
        print(f"  Death: {n_dead:,} final studies")

        valid = ~np.isnan(all_years)
        plot_histogram(
            all_years[death_vals == 1], all_years[valid],
            f"Mortality — final study before death  (n={n_dead})",
            out / "hist_death.png",
            pos_label="Final study (deceased)",
        )

    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
