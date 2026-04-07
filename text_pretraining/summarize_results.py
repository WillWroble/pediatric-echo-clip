"""Summarize eval_supervised results across all runs.

Crawls results/*/eval_supervised/results.csv and produces:
    - summary_heatmap.png   : runs × targets AUROC heatmap
    - summary_best.png      : best AUROC per target across all runs (bar chart)
    - summary_best.csv      : best AUROC per target + which run achieved it
    - summary_death.png     : Death AUROC across all runs (bar chart)
    - summary_full.csv      : full table (runs × targets)

Usage:
    python -u summarize_results.py --results_dir results --output_dir results/summary
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_all(results_dir):
    """Crawl results/*/eval_supervised/results.csv → dict {run_name: Series}."""
    rows = {}
    for csv_path in sorted(Path(results_dir).glob("*/eval_supervised/results.csv")):
        run_name = csv_path.parts[-3]
        df = pd.read_csv(csv_path)
        if "target" not in df.columns or "auroc" not in df.columns:
            print(f"  Skipping {csv_path} (unexpected format)")
            continue
        rows[run_name] = df.set_index("target")["auroc"]
    print(f"Loaded {len(rows)} runs")
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_heatmap(matrix, run_names, targets, path):
    fig, ax = plt.subplots(figsize=(max(10, len(targets) * 0.35), max(6, len(run_names) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(run_names)))
    ax.set_yticklabels(run_names, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="AUROC")
    ax.set_title("AUROC — all runs × all targets  (sorted by Death AUROC)")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_best_bar(best_df, path):
    best_df = best_df.sort_values("auroc", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(best_df) * 0.32)))
    bars = ax.barh(best_df["target"], best_df["auroc"], color="steelblue", alpha=0.8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUROC")
    ax.set_title("Best AUROC per target (across all runs)")
    for bar, (_, row) in zip(bars, best_df.iterrows()):
        ax.text(row["auroc"] + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row['auroc']:.3f}  ({row['run']})", va="center", fontsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_death_bar(death_series, path):
    """death_series: Series indexed by run_name, sorted descending."""
    fig, ax = plt.subplots(figsize=(max(8, len(death_series) * 0.45), 5))
    colors = ["tomato" if v == death_series.max() else "steelblue" for v in death_series.values]
    bars = ax.bar(range(len(death_series)), death_series.values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(death_series)))
    ax.set_xticklabels(death_series.index, rotation=45, ha="right", fontsize=7)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Death AUROC — all runs (sorted descending)")
    for bar, val in zip(bars, death_series.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output_dir",  default="results/summary")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_all(args.results_dir)
    if not rows:
        print("No results found.")
        return

    # Build full table
    full_df = pd.DataFrame(rows).T  # runs × targets
    full_df.index.name = "run"

    # Sort by Death AUROC descending
    if "Death" in full_df.columns:
        full_df = full_df.sort_values("Death", ascending=False)

    full_df.to_csv(out / "summary_full.csv")
    print(f"Saved {out / 'summary_full.csv'}")

    # Heatmap
    targets    = full_df.columns.tolist()
    run_names  = full_df.index.tolist()
    matrix     = full_df.values.astype(float)
    plot_heatmap(matrix, run_names, targets, out / "summary_heatmap.png")

    # Best per target
    best_records = []
    for col in full_df.columns:
        valid = full_df[col].dropna()
        if valid.empty:
            continue
        best_run  = valid.idxmax()
        best_auroc = valid.max()
        best_records.append(dict(target=col, auroc=best_auroc, run=best_run))
    best_df = pd.DataFrame(best_records)
    best_df.sort_values("auroc", ascending=False).to_csv(out / "summary_best.csv", index=False)
    print(f"Saved {out / 'summary_best.csv'}")
    plot_best_bar(best_df, out / "summary_best.png")

    # Death bar
    if "Death" in full_df.columns:
        death_series = full_df["Death"].dropna().sort_values(ascending=False)
        plot_death_bar(death_series, out / "summary_death.png")

    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
