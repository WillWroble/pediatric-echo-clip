"""Eigendecomposition of the trajectory projection head.

Decomposes W from traj_proj into eigenvectors (dynamics axes) and eigenvalues
(progression/recovery/stable modes), then projects embeddings onto top-k
eigenvectors and visualizes with UMAPs.

Usage:
    python -u eigen_analysis.py \
        --checkpoint results/joint_v6/latest.pt \
        --embeddings results/joint_v6/eval/embeddings.npz \
        --top_k 20 \
        --output_dir results/eigen \
        --csv /path/to/echo_reports_v2.csv \
        --labels labels.csv \
        --label_cols LVEF \
        --mask_manifest manifests/subset.txt \
        --max_umap 50000
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from umap import UMAP


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_traj_weights(ckpt_path):
    """Extract W and b from traj_proj in checkpoint."""
    import torch
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = ckpt["encoder_state_dict"]
    W = sd["traj_proj.weight"].numpy()  # (768, 768)
    b = sd["traj_proj.bias"].numpy()    # (768,)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded traj_proj from {ckpt_path} (epoch {epoch})")
    print(f"  W: {W.shape}, b: {b.shape}")
    return W, b


def traj_transform(Z, W, b):
    """Apply the true trajectory head: W @ gelu(z) + b for each row."""
    from scipy.special import erf
    gelu_Z = Z * 0.5 * (1.0 + erf(Z / np.sqrt(2.0)))
    return gelu_Z @ W.T + b[None, :]


def eigen_decompose(W, top_k=20):
    """Eigendecompose W, sort by |λ| descending, return top-k."""
    eigenvalues, eigenvectors = np.linalg.eig(W)
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    print(f"Top-5 |λ|: {np.abs(eigenvalues[:5]).round(4)}")
    print(f"# complex pairs: {(np.abs(eigenvalues.imag) > 1e-6).sum() // 2}")
    return eigenvalues[:top_k], eigenvectors[:, :top_k], eigenvalues


def project(embeddings, eigenvectors):
    """Project embeddings onto eigenvectors. Returns real part."""
    return (embeddings @ eigenvectors).real


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def parse_age_years(age_str, dob_str=None, study_date_str=None):
    """Best-effort age in years from age string or DOB + study_date."""
    if pd.isna(age_str) or str(age_str).strip() == "":
        if dob_str and study_date_str:
            return _age_from_dates(dob_str, study_date_str)
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
        if dob_str and study_date_str:
            return _age_from_dates(dob_str, study_date_str)
        return np.nan


def _age_from_dates(dob_str, sd_str):
    try:
        sd = datetime.strptime(str(sd_str), "%m%d%y%H%M%S")
    except Exception:
        return np.nan
    for fmt in ["%B %d, %Y", "%d-%b-%Y", "%m/%d/%Y"]:
        try:
            dob = datetime.strptime(str(dob_str).strip(), fmt)
            return (sd - dob).days / 365.25
        except Exception:
            continue
    return np.nan


def load_demographics(csv_path, study_ids):
    """Load age, gender, weight, bsa from echo_reports_v2.csv."""
    df = pd.read_csv(csv_path, dtype=str)
    df["study_id"] = df["study_id"].astype(str)
    df = df.drop_duplicates(subset="study_id").set_index("study_id")

    matched = sum(1 for s in study_ids if s in df.index)
    print(f"  Demographics: matched {matched:,}/{len(study_ids):,} studies")

    age = np.array([
        parse_age_years(
            df.loc[s, "age"] if s in df.index else None,
            df.loc[s, "dob"] if s in df.index else None,
            df.loc[s, "study_date"] if s in df.index else None,
        ) for s in study_ids
    ])

    gender = np.full(len(study_ids), np.nan)
    for i, s in enumerate(study_ids):
        if s in df.index:
            g = str(df.loc[s, "gender"]).strip().lower()
            if g == "male":
                gender[i] = 1.0
            elif g == "female":
                gender[i] = 0.0

    def _safe_float(s, col):
        if s not in df.index:
            return np.nan
        v = df.loc[s, col]
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan

    weight = np.array([_safe_float(s, "weight_kg") for s in study_ids])
    bsa = np.array([_safe_float(s, "bsa") for s in study_ids])

    return {"age": age, "gender": gender, "weight_kg": weight, "bsa": bsa}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_eigenspectrum(eigenvalues, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    mags = np.abs(eigenvalues)
    ax1.semilogy(mags, ".-", markersize=3, linewidth=0.8)
    ax1.set_xlabel("Index (sorted by |λ|)")
    ax1.set_ylabel("|λ|")
    ax1.set_title("Eigenvalue Magnitude Spectrum")
    ax1.axhline(1.0, color="gray", ls="--", lw=0.6, alpha=0.5)

    real, imag = eigenvalues.real, eigenvalues.imag
    colors = mags / mags.max()
    sc = ax2.scatter(real, imag, c=colors, cmap="viridis", s=8, alpha=0.7)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_xlabel("Re(λ)")
    ax2.set_ylabel("Im(λ)")
    ax2.set_title("Eigenvalues in Complex Plane")
    ax2.set_aspect("equal")
    plt.colorbar(sc, ax=ax2, label="|λ| (normalized)", shrink=0.8)

    plt.tight_layout()
    path = output_dir / "eigenspectrum.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_umap_colored(coords, values, title, path, cmap="viridis",
                      vmin=None, vmax=None, categorical=False, cat_labels=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    valid = ~np.isnan(values)
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1],
                   s=0.3, alpha=0.08, c="lightgray")
    if categorical:
        for v in np.unique(values[valid]):
            m = valid & (values == v)
            label = cat_labels.get(int(v), str(v)) if cat_labels else str(v)
            ax.scatter(coords[m, 0], coords[m, 1], s=0.5, alpha=0.3, label=label)
        ax.legend(markerscale=10, fontsize=9)
    else:
        vmin = vmin if vmin is not None else np.nanpercentile(values, 2)
        vmax = vmax if vmax is not None else np.nanpercentile(values, 98)
        sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=0.5, alpha=0.3,
                        c=values[valid], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_umap_unlabeled(coords, title, path, c="tab:blue"):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, alpha=0.3, c=c)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_overlay(coords_base, coords_trans, title, path, n_arrows=500, seed=42):
    rng = np.random.RandomState(seed)
    N = coords_base.shape[0]
    arrow_idx = rng.choice(N, min(n_arrows, N), replace=False)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(coords_base[:, 0], coords_base[:, 1],
               s=0.5, alpha=0.15, c="tab:blue", label="Base (Z)")
    ax.scatter(coords_trans[:, 0], coords_trans[:, 1],
               s=0.5, alpha=0.15, c="tab:orange", label="Transformed (traj(Z))")

    for i in arrow_idx:
        ax.annotate("", xy=coords_trans[i], xytext=coords_base[i],
                     arrowprops=dict(arrowstyle="-|>", color="black",
                                     lw=0.4, alpha=0.3, mutation_scale=6))

    ax.legend(markerscale=15, fontsize=10)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_overlay_colored(coords_base, coords_trans, values, label_name,
                         path, n_arrows=500, seed=42):
    rng = np.random.RandomState(seed)
    valid = ~np.isnan(values)
    if valid.sum() < 10:
        return

    vmin, vmax = np.nanpercentile(values, 2), np.nanpercentile(values, 98)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(coords_base[valid, 0], coords_base[valid, 1],
               s=3, alpha=0.3, c=values[valid], cmap="viridis",
               vmin=vmin, vmax=vmax, marker="o", edgecolors="none")
    sc = ax.scatter(coords_trans[valid, 0], coords_trans[valid, 1],
                    s=4, alpha=0.5, c=values[valid], cmap="viridis",
                    vmin=vmin, vmax=vmax, marker="^", edgecolors="none")
    plt.colorbar(sc, ax=ax, shrink=0.8, label=label_name)

    valid_idx = np.where(valid)[0]
    arrow_idx = rng.choice(len(valid_idx), min(n_arrows, len(valid_idx)), replace=False)
    for ai in arrow_idx:
        i = valid_idx[ai]
        ax.annotate("", xy=coords_trans[i], xytext=coords_base[i],
                     arrowprops=dict(arrowstyle="-|>",
                                     color=cmap_obj(norm(values[i])),
                                     lw=0.4, alpha=0.4, mutation_scale=6))

    ax.set_title(f"Base → Trajectory — {label_name}")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_vector_field(coords_base, coords_traj, title, path,
                      values=None, label_name=None,
                      n_arrows=2000, arrow_scale=0.3, seed=42):
    """Base UMAP with normalized arrows showing predicted trajectory direction."""
    rng = np.random.RandomState(seed)
    N = coords_base.shape[0]
    idx = rng.choice(N, min(n_arrows, N), replace=False)

    deltas = coords_traj[idx] - coords_base[idx]
    norms = np.linalg.norm(deltas, axis=1, keepdims=True).clip(1e-8)
    deltas_unit = deltas / norms

    scale = arrow_scale * min(np.ptp(coords_base[:, 0]), np.ptp(coords_base[:, 1]))
    deltas_scaled = deltas_unit * scale

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(coords_base[:, 0], coords_base[:, 1],
               s=0.3, alpha=0.1, c="lightgray", zorder=1)

    if values is not None:
        valid = ~np.isnan(values[idx])
        vmin = np.nanpercentile(values, 2)
        vmax = np.nanpercentile(values, 98)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = matplotlib.colormaps["viridis"]

        for j, i in enumerate(idx):
            if not valid[j]:
                continue
            ax.annotate("", xy=coords_base[i] + deltas_scaled[j],
                         xytext=coords_base[i],
                         arrowprops=dict(arrowstyle="-|>",
                                         color=cmap_obj(norm(values[i])),
                                         lw=0.6, alpha=0.5, mutation_scale=7),
                         zorder=3)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                     ax=ax, shrink=0.8, label=label_name or "")
    else:
        for j, i in enumerate(idx):
            ax.annotate("", xy=coords_base[i] + deltas_scaled[j],
                         xytext=coords_base[i],
                         arrowprops=dict(arrowstyle="-|>", color="tab:red",
                                         lw=0.5, alpha=0.4, mutation_scale=7),
                         zorder=3)

    suffix = f" — {label_name}" if label_name else ""
    ax.set_title(f"{title}{suffix}")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_eigenvector_projections(coords, study_ids, labels_dict,
                                eigenvalues, output_dir, top_pairs=3):
    for label_name, label_vals in labels_dict.items():
        vals = np.array([label_vals.get(sid, np.nan) for sid in study_ids])
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            print(f"  Skipping {label_name}: only {valid.sum()} matched")
            continue

        vmin, vmax = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
        for i in range(min(top_pairs, coords.shape[1] - 1)):
            fig, ax = plt.subplots(figsize=(8, 7))
            if (~valid).any():
                ax.scatter(coords[~valid, i], coords[~valid, i + 1],
                           s=0.5, alpha=0.1, c="lightgray")
            sc = ax.scatter(coords[valid, i], coords[valid, i + 1],
                            s=2, alpha=0.4, c=vals[valid], cmap="viridis",
                            vmin=vmin, vmax=vmax)
            plt.colorbar(sc, ax=ax, shrink=0.8, label=label_name)
            ax.set_xlabel(f"EV {i} (|λ|={np.abs(eigenvalues[i]):.3f})")
            ax.set_ylabel(f"EV {i+1} (|λ|={np.abs(eigenvalues[i+1]):.3f})")
            ax.set_title(f"Eigenvector Projection — {label_name}")
            plt.tight_layout()
            path = output_dir / f"proj_ev{i}_ev{i+1}_{label_name}.png"
            plt.savefig(path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_labels(labels_path, label_cols):
    df = pd.read_csv(labels_path)
    sid_col = None
    for c in ["study_id", "Event.ID.Number", "sid", "eid"]:
        if c in df.columns:
            sid_col = c
            break
    if sid_col is None:
        raise ValueError(f"No study ID column found in {labels_path}")

    df["_sid"] = df[sid_col].astype(int).astype(str)
    labels_dict = {}
    for col in label_cols:
        if col not in df.columns:
            print(f"  Warning: {col} not in labels CSV, skipping")
            continue
        sub = df[["_sid", col]].dropna(subset=[col])
        labels_dict[col] = dict(zip(sub["_sid"], sub[col].astype(float)))
        print(f"  Loaded {len(labels_dict[col]):,} labels for {col}")
    return labels_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Eigen analysis of trajectory head")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", required=True, help=".npz from eval.py")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--top_k", type=int, default=20)

    p.add_argument("--csv", default=None,
                   help="echo_reports_v2.csv for demographics coloring")
    p.add_argument("--labels", default=None, help="CSV with clinical labels")
    p.add_argument("--label_cols", default=None, help="Comma-sep columns")

    p.add_argument("--mask_manifest", default=None,
                   help="Text file of study_ids to include (one per line)")
    p.add_argument("--max_umap", type=int, default=None,
                   help="Max studies per UMAP (subsample if exceeded)")

    p.add_argument("--n_arrows", type=int, default=500)
    p.add_argument("--n_field_arrows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load weights ----
    W, b = load_traj_weights(args.checkpoint)

    # ---- Load embeddings ----
    emb_data = np.load(args.embeddings)
    Z = emb_data[args.split]
    study_ids = emb_data[f"{args.split}_ids"].tolist()
    print(f"Loaded {len(study_ids):,} studies, {Z.shape[1]}d ({args.split})")

    # ---- Mask manifest ----
    if args.mask_manifest:
        mask_ids = set(
            Path(args.mask_manifest).read_text().strip().splitlines()
        )
        keep = [i for i, s in enumerate(study_ids) if s in mask_ids]
        Z = Z[keep]
        study_ids = [study_ids[i] for i in keep]
        print(f"After mask manifest: {len(study_ids):,} studies")

    # ---- Eigendecompose ----
    top_vals, top_vecs, all_vals = eigen_decompose(W, args.top_k)

    # ---- Eigenvector projections ----
    ev_coords = project(Z, top_vecs)

    # ---- True trajectory transform ----
    ZT = traj_transform(Z, W, b)

    # ---- Collect coloring sources ----
    color_fields = {}

    if args.csv:
        print("\nLoading demographics...")
        demos = load_demographics(args.csv, study_ids)
        color_fields["age"] = (demos["age"], dict(cmap="plasma", vmin=0, vmax=25))
        color_fields["gender"] = (demos["gender"],
                                  dict(categorical=True,
                                       cat_labels={0: "Female", 1: "Male"}))
        color_fields["weight_kg"] = (demos["weight_kg"], dict(cmap="viridis"))
        color_fields["bsa"] = (demos["bsa"], dict(cmap="viridis"))

    labels_dict = {}
    if args.labels and args.label_cols:
        label_cols = [c.strip() for c in args.label_cols.split(",")]
        labels_dict = load_labels(args.labels, label_cols)
        for name, vals in labels_dict.items():
            arr = np.array([vals.get(s, np.nan) for s in study_ids])
            color_fields[name] = (arr, dict(cmap="viridis"))

    # ---- Subsample for UMAPs ----
    rng = np.random.RandomState(args.seed)
    N = Z.shape[0]
    cap = args.max_umap or N
    if cap < N:
        idx = rng.choice(N, cap, replace=False)
        print(f"\nSubsampling {cap:,}/{N:,} for UMAPs")
    else:
        idx = np.arange(N)
    Z_sub = Z[idx]
    ZT_sub = ZT[idx]
    ids_sub = [study_ids[i] for i in idx]
    color_sub = {k: (v[idx], kw) for k, (v, kw) in color_fields.items()}

    # ---- Plots ----
    print("\n--- Eigenspectrum ---")
    plot_eigenspectrum(all_vals, out)

    print("\n--- Eigenvector projections ---")
    if labels_dict:
        plot_eigenvector_projections(ev_coords, study_ids, labels_dict, top_vals, out)

    # UMAP 1: base space
    print("\n--- UMAP: base space ---")
    umap_base = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                     random_state=args.seed)
    coords_base = umap_base.fit_transform(Z_sub)
    plot_umap_unlabeled(coords_base, "Base Embedding Space (Z)",
                        out / "umap_base.png")
    for name, (vals, kw) in color_sub.items():
        plot_umap_colored(coords_base, vals, f"Base Space — {name}",
                          out / f"umap_base_{name}.png", **kw)

    # UMAP 2: trajectory space
    print("\n--- UMAP: trajectory space ---")
    umap_traj = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                     random_state=args.seed)
    coords_traj = umap_traj.fit_transform(ZT_sub)
    plot_umap_unlabeled(coords_traj, "Trajectory Space (traj(Z))",
                        out / "umap_traj.png", c="tab:orange")
    for name, (vals, kw) in color_sub.items():
        plot_umap_colored(coords_traj, vals, f"Trajectory Space — {name}",
                          out / f"umap_traj_{name}.png", **kw)

    # UMAP 3: overlay — joint fit
    print("\n--- UMAP: overlay ---")
    joint = np.concatenate([Z_sub, ZT_sub], axis=0)
    umap_joint = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                      random_state=args.seed)
    coords_joint = umap_joint.fit_transform(joint)
    coords_j_base = coords_joint[:len(Z_sub)]
    coords_j_trans = coords_joint[len(Z_sub):]

    plot_overlay(coords_j_base, coords_j_trans,
                 "Base → Trajectory Transformation",
                 out / "umap_overlay.png",
                 n_arrows=args.n_arrows, seed=args.seed)

    for name, (vals, kw) in color_sub.items():
        if kw.get("categorical"):
            continue
        plot_overlay_colored(coords_j_base, coords_j_trans, vals, name,
                             out / f"umap_overlay_{name}.png",
                             n_arrows=args.n_arrows, seed=args.seed)

    # UMAP 4: vector field — base space with trajectory direction arrows
    print("\n--- UMAP: vector field ---")
    plot_vector_field(
        coords_j_base, coords_j_trans,
        "Predicted Trajectory Field", out / "umap_field.png",
        n_arrows=args.n_field_arrows, seed=args.seed,
    )
    for name, (vals, kw) in color_sub.items():
        if kw.get("categorical"):
            continue
        plot_vector_field(
            coords_j_base, coords_j_trans,
            "Predicted Trajectory Field", out / f"umap_field_{name}.png",
            values=vals, label_name=name,
            n_arrows=args.n_field_arrows, seed=args.seed,
        )

    # ---- Save artifacts ----
    np.save(out / "eigenvalues.npy", all_vals)
    np.save(out / "eigenvectors_top.npy", top_vecs)
    np.savez_compressed(out / "projections.npz",
                        coords=ev_coords, study_ids=study_ids)

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_studies": int(Z.shape[0]),
        "n_umap": int(len(Z_sub)),
        "top_k": args.top_k,
        "top_eigenvalues": [
            {"rank": i, "magnitude": round(float(np.abs(v)), 6),
             "real": round(float(v.real), 6), "imag": round(float(v.imag), 6)}
            for i, v in enumerate(top_vals)
        ],
        "n_complex_pairs": int((np.abs(all_vals.imag) > 1e-6).sum() // 2),
        "spectral_norm": round(float(np.abs(all_vals[0])), 6),
        "trace": round(float(np.trace(W)), 6),
        "frobenius_norm": round(float(np.linalg.norm(W, "fro")), 6),
        "color_fields": list(color_fields.keys()),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out / 'summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
