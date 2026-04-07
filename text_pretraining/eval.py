"""Unsupervised UMAP visualizations for report and/or study embeddings.

Usage:
    python -u eval.py \
        --report_embeddings results/v1/eval/embeddings.npz \
        --study_embeddings  results/v1/eval/embeddings_video.npz \
        --h5_dir /lab-share/.../Line_Embeddings \
        --labels Echo_Labels_SG_Fyler_112025.csv \
        --death_mrn death_mrn.csv \
        --output_dir results/v1/eval
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from umap import UMAP

from report_dataset import preload_all, parse_study_date


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_embeddings(path):
    """Load .npz → (Z_train, train_ids, Z_val, val_ids).
    Old format (embeddings/study_ids): Z_train=None, train_ids=None."""
    npz = np.load(path)
    if "train" in npz:
        return (npz["train"].astype(np.float32), npz["train_ids"].astype(str).tolist(),
                npz["val"].astype(np.float32),   npz["val_ids"].astype(str).tolist())
    return (None, None,
            npz["embeddings"].astype(np.float32), npz["study_ids"].astype(str).tolist())


def load_label_info(labels_path):
    """Load Fyler CSV → (eid_to_pid, patient_labels, diag_cols)."""
    label_df = pd.read_csv(labels_path, encoding="utf-8-sig")
    label_df["eid"] = label_df["eid"].astype(str)
    label_df["pid"] = label_df["pid"].astype(str)

    diag_cols = [c for c in [
        "Composite_Critical", "Composite_NonCritical",
        "VSD", "ASD", "TOF", "AVCD", "Coarct", "HLHS", "PDA",
        "TOF_ADJ", "AVCD_ADJ", "VSD_ADJ", "Coarct_ADJ",
    ] if c in label_df.columns]

    patient_labels = label_df.groupby("pid")[diag_cols].max()
    eid_to_pid = dict(zip(label_df["eid"], label_df["pid"]))
    return eid_to_pid, patient_labels, diag_cols


def load_death_mrns(death_mrn_path):
    """Load death MRN CSV → set of normalized MRNs."""
    death_df = pd.read_csv(death_mrn_path)
    death_df["mrn"] = death_df["mrn"].astype(str)
    return set(death_df["mrn"].str.lstrip("0"))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def vicreg_stats(Z):
    Z = torch.from_numpy(Z).float()
    N, D = Z.shape
    stds = Z.std(dim=0)
    Z_c = Z - Z.mean(dim=0)
    cov = (Z_c.T @ Z_c) / (N - 1)
    std_outer = (stds.unsqueeze(0) * stds.unsqueeze(1)).clamp(min=1e-8)
    corr = cov / std_outer
    off = corr[~torch.eye(D, dtype=torch.bool)]
    return {
        "var_mean": round(stds.mean().item(), 4),
        "var_min": round(stds.min().item(), 4),
        "collapsed_dims": int((stds < 0.01).sum()),
        "cov_off_diag_mean": round(off.abs().mean().item(), 4),
        "cov_off_diag_max": round(off.abs().max().item(), 4),
    }


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def build_meta(val_ids, data):
    meta = []
    for sid in val_ids:
        lines, demos, mrn, sd_str = data[sid]
        meta.append(dict(
            study_id=sid, mrn=mrn, age=demos[0], gender=demos[1],
            weight_kg=demos[2], height_cm=demos[3],
            bsa=demos[4], bmi=demos[5], n_lines=lines.shape[0],
            study_date=parse_study_date(sd_str),
        ))
    return meta


def to_fractional_year(dt):
    if dt is None:
        return np.nan
    return dt.year + (dt.timetuple().tm_yday - 1) / 365.25


def label_values(val_ids, col, eid_to_pid, patient_labels):
    """Map study IDs → binary label array via eid→pid→patient_labels."""
    vals = np.full(len(val_ids), np.nan)
    for i, sid in enumerate(val_ids):
        pid = eid_to_pid.get(sid)
        if pid is not None and pid in patient_labels.index and pd.notna(patient_labels.loc[pid, col]):
            vals[i] = float(patient_labels.loc[pid, col])
    return vals


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
        vmin = vmin or np.nanpercentile(values, 2)
        vmax = vmax or np.nanpercentile(values, 98)
        sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=0.5, alpha=0.3,
                        c=values[valid], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_plain(coords, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, alpha=0.3, c="steelblue")
    ax.set_title(f"UMAP (n={len(coords):,})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_landmarks(coords, all_ids, data, path, k=15, seed=42):
    n = len(coords)
    selected = [np.random.RandomState(seed).randint(n)]
    dists = np.full(n, np.inf)
    for _ in range(k - 1):
        last = coords[selected[-1]]
        dists = np.minimum(dists, np.linalg.norm(coords - last, axis=1))
        selected.append(np.argmax(dists))

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.3, alpha=0.08, c="lightgray")
    cmap_k = cm.get_cmap("tab20", k)
    for i, idx in enumerate(selected):
        color = cmap_k(i)
        sid = all_ids[idx]
        _, _, mrn, _ = data[sid]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=80, c=[color],
                   edgecolors="k", linewidths=0.8, zorder=5, label=mrn or sid)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
              markerscale=1, frameon=True, title="MRN")
    ax.set_title(f"{k} Landmark Studies")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_trajectories(coords, all_ids, data, path, n_patients=5, seed=42):
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    patient_studies = defaultdict(list)
    for sid in all_ids:
        _, _, mrn, sd_str = data[sid]
        dt = parse_study_date(sd_str)
        if mrn and dt:
            patient_studies[mrn].append((dt, sid))
    for mrn in patient_studies:
        patient_studies[mrn].sort()

    exact = [(mrn, s) for mrn, s in patient_studies.items() if len(s) == 10]
    if len(exact) >= n_patients:
        rng = np.random.RandomState(seed)
        rng.shuffle(exact)
        top = exact[:n_patients]
    else:
        ranked = sorted(patient_studies.items(), key=lambda x: (abs(len(x[1]) - 10), -len(x[1])))
        rng = np.random.RandomState(seed)
        rng.shuffle(ranked)
        ranked.sort(key=lambda x: abs(len(x[1]) - 10))
        top = ranked[:n_patients]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.3, alpha=0.08, c="lightgray")
    cmap_t = cm.get_cmap("tab20", n_patients)
    for i, (mrn, studies) in enumerate(top):
        idxs = [id_to_idx[sid] for _, sid in studies if sid in id_to_idx]
        if len(idxs) < 2:
            continue
        pts = coords[idxs]
        color = cmap_t(i)
        for j in range(len(pts) - 1):
            ax.annotate("", xy=pts[j + 1], xytext=pts[j],
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                        mutation_scale=10), zorder=4)
        ax.scatter(pts[:, 0], pts[:, 1], s=12, c=[color], zorder=5,
                   edgecolors="k", linewidths=0.3)
        ax.scatter(pts[0, 0], pts[0, 1], s=50, c=[color], marker="o",
                   edgecolors="k", linewidths=1.0, zorder=6)
        ax.plot([], [], color=color, marker="o", ls="-", markersize=5,
                label=f"{mrn} ({len(studies)})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
              markerscale=1, frameon=True, title="MRN (visits)")
    ax.set_title(f"{n_patients} Patient Trajectories (~10 visits)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_dx_trajectories(coords, all_ids, meta, data, patient_labels,
                         mrn_to_pid, dx, path, n_patients=10):
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    pos_pids = set(patient_labels.index[patient_labels[dx] == 1])
    patient_studies = defaultdict(list)
    for i, sid in enumerate(all_ids):
        mrn = meta[i]["mrn"]
        pid = mrn_to_pid.get(mrn)
        if pid not in pos_pids:
            continue
        _, _, _, sd_str = data[sid]
        dt = parse_study_date(sd_str)
        if mrn and dt:
            patient_studies[mrn].append((dt, sid))
    for mrn in patient_studies:
        patient_studies[mrn].sort()

    patient_studies = {m: s for m, s in patient_studies.items() if len(s) >= 2}
    if len(patient_studies) < n_patients:
        print(f"  {dx}: only {len(patient_studies)} positive patients with 2+ studies")
        n_patients = max(len(patient_studies), 1)

    mrn_list = list(patient_studies.keys())
    centroids = np.array([
        coords[[id_to_idx[sid] for _, sid in patient_studies[mrn] if sid in id_to_idx]].mean(axis=0)
        for mrn in mrn_list
    ])
    chosen = np.argsort(np.linalg.norm(centroids - centroids.mean(axis=0), axis=1))[:n_patients]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.3, alpha=0.08, c="lightgray")
    cmap_t = cm.get_cmap("tab20", n_patients)
    for i, ci in enumerate(chosen):
        mrn = mrn_list[ci]
        studies = patient_studies[mrn]
        idxs = [id_to_idx[sid] for _, sid in studies if sid in id_to_idx]
        if len(idxs) < 2:
            continue
        pts = coords[idxs]
        color = cmap_t(i)
        for j in range(len(pts) - 1):
            ax.annotate("", xy=pts[j + 1], xytext=pts[j],
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                        mutation_scale=10), zorder=4)
        ax.scatter(pts[:, 0], pts[:, 1], s=12, c=[color], zorder=5,
                   edgecolors="k", linewidths=0.3)
        ax.scatter(pts[0, 0], pts[0, 1], s=50, c=[color], marker="o",
                   edgecolors="k", linewidths=1.0, zorder=6)
        ax.plot([], [], color=color, marker="o", ls="-", markersize=5,
                label=f"{mrn} ({len(studies)})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
              markerscale=1, frameon=True, title="MRN (visits)")
    ax.set_title(f"{dx}: {n_patients} Tightest-Clustering Patients")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


def plot_variance_spectrum(Z_train, Z_val, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    if Z_train is not None:
        ax.plot(np.sort(np.std(Z_train, axis=0))[::-1], label="train", color="tab:blue", alpha=0.8)
    ax.plot(np.sort(np.std(Z_val, axis=0))[::-1], label="val", color="tab:orange", alpha=0.8)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="VICReg target")
    ax.set_xlabel("Dimension (sorted)"); ax.set_ylabel("Std")
    ax.set_title("Variance Spectrum"); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Per-side evaluation
# ---------------------------------------------------------------------------

def eval_side(Z_train, train_ids, Z_val, val_ids, out, data=None,
              label_info=None, dead_mrns=None,
              n_landmarks=15, n_trajectory_patients=5):
    """Run all unsupervised eval for one embedding side."""
    out.mkdir(parents=True, exist_ok=True)
    has_train = Z_train is not None

    # Filter against h5 if present
    if data is not None:
        if has_train:
            mask = [i for i, s in enumerate(train_ids) if s in data]
            Z_train, train_ids = Z_train[mask], [train_ids[i] for i in mask]
        mask = [i for i, s in enumerate(val_ids) if s in data]
        Z_val, val_ids = Z_val[mask], [val_ids[i] for i in mask]

    print(f"  {'Train: ' + f'{len(train_ids):,}  ' if has_train else ''}Val: {len(val_ids):,}", flush=True)

    # Stats
    stats = {}
    if has_train:
        stats["train"] = vicreg_stats(Z_train)
        print("  --- Train ---")
        for k, v in stats["train"].items():
            print(f"    {k}: {v}")
    stats["val"] = vicreg_stats(Z_val)
    print("  --- Val ---")
    for k, v in stats["val"].items():
        print(f"    {k}: {v}")
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Variance spectrum
    plot_variance_spectrum(Z_train, Z_val, out / "variance_spectrum.png")

    # UMAP (val only)
    print("  Fitting UMAP (val only)...", flush=True)
    coords = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42).fit_transform(Z_val)
    plot_plain(coords, out / "umap.png")

    # Metadata UMAPs (need h5)
    meta_val = None
    if data is not None:
        meta_val = build_meta(val_ids, data)
        plot_colored(coords, np.array([m["age"] for m in meta_val]),
                     "Age (years)", out / "umap_age.png", cmap="plasma", vmin=0, vmax=25)
        plot_colored(coords, np.array([m["gender"] for m in meta_val]),
                     "Gender", out / "umap_gender.png",
                     cmap="coolwarm", categorical=True, cat_labels={0: "Female", 1: "Male"})
        plot_colored(coords, np.array([m["weight_kg"] for m in meta_val]),
                     "Weight (kg)", out / "umap_weight.png")
        plot_colored(coords, np.array([m["bsa"] for m in meta_val]),
                     "BSA (m²)", out / "umap_bsa.png")
        plot_colored(coords, np.array([m["n_lines"] for m in meta_val], dtype=np.float64),
                     "Report Lines", out / "umap_nlines.png", cmap="inferno")
        plot_colored(coords, np.array([to_fractional_year(m["study_date"]) for m in meta_val]),
                     "Study Date", out / "umap_study_date.png", cmap="viridis")

    # Diagnostic label UMAPs (eid→pid join, no h5 needed)
    if label_info is not None:
        eid_to_pid, patient_labels, diag_cols = label_info
        print("  Plotting diagnostic labels...", flush=True)

        for col in diag_cols:
            vals = label_values(val_ids, col, eid_to_pid, patient_labels)
            n_pos = int(np.nansum(vals == 1))
            plot_colored(coords, vals, f"{col} (n={n_pos})",
                         out / f"umap_dx_{col.lower()}.png", cmap="coolwarm",
                         categorical=True, cat_labels={0: "Negative", 1: "Positive"})

        # Dx trajectory plots (need h5 for mrn→pid bridge and study dates)
        if data is not None:
            mrn_to_pid = {}
            for sid in val_ids:
                pid = eid_to_pid.get(sid)
                if pid is not None and sid in data:
                    _, _, mrn, _ = data[sid]
                    if mrn:
                        mrn_to_pid[mrn] = pid

            for dx in ["TOF_ADJ", "HLHS", "AVCD_ADJ"]:
                if dx not in patient_labels.columns:
                    continue
                plot_dx_trajectories(
                    coords, val_ids, meta_val, data, patient_labels,
                    mrn_to_pid, dx, out / f"umap_dx_traj_{dx.lower()}.png", n_patients=10,
                )

    # Death UMAP (needs h5 + death_mrn)
    if dead_mrns is not None and data is not None and meta_val is not None:
        print("  Plotting death overlay...", flush=True)
        patient_latest = {}
        for i, m in enumerate(meta_val):
            mrn_norm = m["mrn"].lstrip("0")
            if mrn_norm not in dead_mrns:
                continue
            dt = m["study_date"]
            if dt is None:
                continue
            if mrn_norm not in patient_latest or dt > patient_latest[mrn_norm][0]:
                patient_latest[mrn_norm] = (dt, i)

        death_vals = np.zeros(len(val_ids))
        for _, idx in patient_latest.values():
            death_vals[idx] = 1.0
        n_dead = len(patient_latest)
        print(f"    {n_dead:,} deceased patients with studies in val set")

        fig, ax = plt.subplots(figsize=(8, 7))
        alive_mask = death_vals == 0
        dead_mask  = death_vals == 1
        ax.scatter(coords[alive_mask, 0], coords[alive_mask, 1],
                   s=0.3, alpha=0.08, c="lightgray")
        ax.scatter(coords[dead_mask, 0], coords[dead_mask, 1],
                   s=8, alpha=0.7, c="red", edgecolors="darkred",
                   linewidths=0.2, zorder=5, label=f"Deceased (n={n_dead})")
        ax.legend(markerscale=2, fontsize=9)
        ax.set_title(f"Deceased — Last Study (n={n_dead})")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(out / "umap_death.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out / 'umap_death.png'}")

    # Landmarks + trajectories (need h5)
    if data is not None:
        for i, seed in enumerate([42, 123, 7]):
            plot_landmarks(coords, val_ids, data, out / f"umap_landmarks_{i+1}.png",
                           k=n_landmarks, seed=seed)
        for i, seed in enumerate([42, 123, 7]):
            plot_trajectories(coords, val_ids, data, out / f"umap_trajectories_{i+1}.png",
                              n_patients=n_trajectory_patients, seed=seed)

    print(f"  Done → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Unsupervised eval for report/study embeddings")
    p.add_argument("--report_embeddings", default=None, help=".npz (new format: train/val/train_ids/val_ids)")
    p.add_argument("--study_embeddings",  default=None, help=".npz (new or old format)")
    p.add_argument("--h5_dir",            default=None, help="Line embeddings HDF5 dir (optional)")
    p.add_argument("--labels",            default=None, help="Fyler diagnostic labels CSV")
    p.add_argument("--death_mrn",         default=None, help="Death MRN CSV")
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--n_landmarks",           type=int, default=15)
    p.add_argument("--n_trajectory_patients", type=int, default=5)
    args = p.parse_args()

    assert args.report_embeddings or args.study_embeddings, \
        "At least one of --report_embeddings / --study_embeddings required"

    out = Path(args.output_dir)

    # Load shared resources
    data = preload_all(args.h5_dir) if args.h5_dir else None
    label_info = load_label_info(args.labels) if args.labels else None
    dead_mrns = load_death_mrns(args.death_mrn) if args.death_mrn else None

    if args.death_mrn and not args.h5_dir:
        print("Warning: --death_mrn requires --h5_dir for MRN mapping, death plots will be skipped", flush=True)

    side_kwargs = dict(data=data, label_info=label_info, dead_mrns=dead_mrns,
                       n_landmarks=args.n_landmarks,
                       n_trajectory_patients=args.n_trajectory_patients)

    if args.report_embeddings:
        print("\n=== Report side ===", flush=True)
        Z_train, train_ids, Z_val, val_ids = load_embeddings(args.report_embeddings)
        eval_side(Z_train, train_ids, Z_val, val_ids, out / "report", **side_kwargs)

    if args.study_embeddings:
        print("\n=== Study side ===", flush=True)
        Z_train, train_ids, Z_val, val_ids = load_embeddings(args.study_embeddings)
        eval_side(Z_train, train_ids, Z_val, val_ids, out / "study", **side_kwargs)

    print(f"\nAll done. Results in {out}")


if __name__ == "__main__":
    main()
