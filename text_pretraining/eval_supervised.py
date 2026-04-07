"""Linear probe evaluation for report and/or study embeddings.

Fits logistic regression probes on frozen embeddings and reports AUROC
for all binary diagnostic targets. Supports loading pre-fit weights.

Usage:
    python -u eval_supervised.py \
        --report_embeddings results/v1/eval/embeddings.npz \
        --study_embeddings  results/v1/eval/embeddings_video.npz \
        --h5_dir /lab-share/.../Line_Embeddings \
        --labels Echo_Labels_SG_Fyler_112025.csv \
        --death_mrn death_mrn.csv \
        --output_dir results/v1/eval_supervised
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
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

    diag_cols = [c for c in label_df.columns
                 if c not in {"eid", "pid", "Gender", "Age"}
                 and not c.strip().lstrip("-").isdigit()]

    patient_labels = label_df.groupby("pid")[diag_cols].max()
    eid_to_pid = dict(zip(label_df["eid"], label_df["pid"]))
    return eid_to_pid, patient_labels, diag_cols


def load_death_mrns(death_mrn_path):
    death_df = pd.read_csv(death_mrn_path)
    death_df["mrn"] = death_df["mrn"].astype(str)
    return set(death_df["mrn"].str.lstrip("0"))


def load_probe_weights(path):
    """Load probe_weights.npz → dict with scaler params + per-target coef/intercept."""
    return dict(np.load(path, allow_pickle=True))


# ---------------------------------------------------------------------------
# Label vectors
# ---------------------------------------------------------------------------

def build_label_matrix(ids, diag_cols, eid_to_pid, patient_labels):
    Y = np.full((len(ids), len(diag_cols)), np.nan, dtype=np.float32)
    for i, sid in enumerate(ids):
        pid = eid_to_pid.get(sid)
        if pid is None or pid not in patient_labels.index:
            continue
        Y[i] = patient_labels.loc[pid, diag_cols].values.astype(np.float32)
    return Y


def build_death_vector(ids, meta, dead_mrns):
    """1 at last study of each deceased patient, 0 elsewhere."""
    patient_latest = {}
    for i, m in enumerate(meta):
        mrn_norm = m["mrn"].lstrip("0")
        if mrn_norm not in dead_mrns or m["study_date"] is None:
            continue
        if mrn_norm not in patient_latest or m["study_date"] > patient_latest[mrn_norm][0]:
            patient_latest[mrn_norm] = (m["study_date"], i)
    vals = np.zeros(len(ids), dtype=np.float32)
    for _, idx in patient_latest.values():
        vals[idx] = 1.0
    return vals, len(patient_latest)


def build_meta(ids, data):
    meta = []
    for sid in ids:
        lines, demos, mrn, sd_str = data[sid]
        meta.append(dict(mrn=mrn, study_date=parse_study_date(sd_str)))
    return meta


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def fit_probe(X_tr, y_tr, X_va, y_va):
    """Fit LogisticRegression → (auroc, fpr, tpr, scores, coef, intercept) or Nones."""
    if y_tr.sum() < 2 or y_va.sum() < 2:
        return None, None, None, None, None, None
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    scores = clf.predict_proba(X_va)[:, 1]
    auroc = roc_auc_score(y_va, scores)
    fpr, tpr, _ = roc_curve(y_va, scores)
    return auroc, fpr, tpr, scores, clf.coef_[0], clf.intercept_[0]


def score_with_weights(X_va, y_va, coef, intercept):
    """Score val set with pre-fit weights → (auroc, fpr, tpr, scores) or Nones."""
    if y_va.sum() < 2:
        return None, None, None, None
    logits = X_va @ coef + intercept
    scores = 1.0 / (1.0 + np.exp(-logits))
    auroc = roc_auc_score(y_va, scores)
    fpr, tpr, _ = roc_curve(y_va, scores)
    return auroc, fpr, tpr, scores
def load_lvef_labels(path):
    """Load ECGLVEF CSV → {study_id_str: lvef_float}."""
    df = pd.read_csv(path)
    df["sid"] = df["Event.ID.Number"].astype(int).astype(str)
    df = df[["sid", "LVEF"]].dropna(subset=["LVEF"]).drop_duplicates(subset="sid")
    return dict(zip(df["sid"], df["LVEF"].astype(np.float32)))


def build_lvef_vector(ids, lvef_map):
    """Map study IDs → LVEF array (NaN where missing)."""
    vals = np.full(len(ids), np.nan, dtype=np.float32)
    for i, sid in enumerate(ids):
        if sid in lvef_map:
            vals[i] = lvef_map[sid]
    return vals


def fit_lvef_probe(X_tr, y_tr, X_va, y_va):
    """Ridge regression → (mae, predictions, coef, intercept) or Nones."""
    if len(y_tr) < 10 or len(y_va) < 10:
        return None, None, None, None
    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_va)
    mae = float(np.abs(preds - y_va).mean())
    return mae, preds, reg.coef_, reg.intercept_


def score_lvef_with_weights(X_va, y_va, coef, intercept):
    """Score val set with pre-fit Ridge weights → (mae, predictions) or Nones."""
    if len(y_va) < 10:
        return None, None
    preds = X_va @ coef + intercept
    mae = float(np.abs(preds - y_va).mean())
    return mae, preds

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_auroc_bar(results, path):
    results = sorted(results, key=lambda r: r["auroc"], reverse=True)
    labels = [r["target"] for r in results]
    aurocs = [r["auroc"]  for r in results]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.32)))
    bars = ax.barh(labels, aurocs, color="steelblue", alpha=0.8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUROC")
    ax.set_title("Linear Probe AUROC")
    for bar, val in zip(bars, aurocs):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_roc_curves(results, path):
    results = sorted(results, key=lambda r: r["auroc"], reverse=True)
    cmap = plt.cm.get_cmap("tab20", len(results))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    for i, r in enumerate(results):
        ax.plot(r["fpr"], r["tpr"], color=cmap(i), linewidth=0.8,
                label=f"{r['target']} ({r['auroc']:.3f})")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — Linear Probe")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_probe_umap(coords, y_true, scores, target, auroc, path):
    """Plot only ground-truth positives, colored by predicted probability."""
    pos_mask = y_true == 1
    if pos_mask.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1], s=2, alpha=0.5,
                    c=scores[pos_mask], cmap="coolwarm", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="P(positive)")
    n_pos = int(pos_mask.sum())
    ax.set_title(f"{target} — Positives Only (n={n_pos}, AUROC={auroc:.3f})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")

def plot_lvef_umap(coords, y_true, preds, mae, path_pred, path_err):
    """Two plots: predicted LVEF coloring + absolute error coloring."""
    valid = ~np.isnan(y_true) & ~np.isnan(preds)
    if valid.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=1, alpha=0.4,
                    c=preds[valid], cmap="viridis", vmin=0, vmax=100)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Predicted LVEF")
    ax.set_title(f"Predicted LVEF (MAE={mae:.2f})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path_pred, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path_pred.name}")

    errors = np.full(len(y_true), np.nan)
    errors[valid] = np.abs(preds[valid] - y_true[valid])

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=1, alpha=0.4,
                    c=errors[valid], cmap="hot_r", vmin=0,
                    vmax=np.nanpercentile(errors[valid], 95))
    plt.colorbar(sc, ax=ax, shrink=0.8, label="|Error|")
    ax.set_title(f"LVEF Absolute Error (MAE={mae:.2f})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path_err, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path_err.name}")

def plot_comparison(report_results, study_results, out):
    """Grouped bar chart comparing report vs study AUROCs."""
    r_map = {r["target"]: r["auroc"] for r in report_results}
    s_map = {r["target"]: r["auroc"] for r in study_results}
    targets = sorted(set(r_map) | set(s_map),
                     key=lambda t: max(r_map.get(t, 0), s_map.get(t, 0)), reverse=True)

    rows = [{"target": t, "report_auroc": r_map.get(t), "study_auroc": s_map.get(t)}
            for t in targets]
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    print(f"Saved {out / 'comparison.csv'}")

    r_vals = [r_map.get(t, 0) for t in targets]
    s_vals = [s_map.get(t, 0) for t in targets]

    x = np.arange(len(targets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, max(4, len(targets) * 0.35)))
    ax.barh(x + w / 2, r_vals, w, label="Report", color="steelblue", alpha=0.8)
    ax.barh(x - w / 2, s_vals, w, label="Study",  color="coral",     alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(targets, fontsize=7)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUROC")
    ax.set_title("Report vs Study — Linear Probe AUROC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out / 'comparison.png'}")


# ---------------------------------------------------------------------------
# Per-side evaluation
# ---------------------------------------------------------------------------

def eval_side(Z_train, train_ids, Z_val, val_ids, out,
              label_info=None, dead_mrns=None, data=None,
              probe_weights_path=None, lvef_map=None):
    """Run supervised eval for one embedding side. Returns list of result dicts."""
    out.mkdir(parents=True, exist_ok=True)
    has_train = Z_train is not None
    using_weights = probe_weights_path is not None

    # Filter against h5 if present (only for metadata, don't drop studies without h5)
    # We need h5-filtered IDs for death only; label join uses eid_to_pid directly.
    # But we DO want to filter for death meta building, so build meta on the h5-present subset.

    print(f"  {'Train: ' + f'{len(train_ids):,}  ' if has_train else ''}Val: {len(val_ids):,}", flush=True)

    # Determine mode
    if using_weights:
        print("  Mode: eval-only (loading pre-fit weights)", flush=True)
    elif has_train:
        print("  Mode: fit probes (train → val)", flush=True)
    else:
        print("  Mode: no train split, no weights → skipping probes", flush=True)
        return []

    # Scale embeddings
    if using_weights:
        weights = load_probe_weights(probe_weights_path)
        scaler_mean = weights["scaler_mean"]
        scaler_scale = weights["scaler_scale"]
        Z_val_scaled = (Z_val - scaler_mean) / scaler_scale
    else:
        scaler = StandardScaler()
        Z_train_scaled = scaler.fit_transform(Z_train)
        Z_val_scaled = scaler.transform(Z_val)

    results = []
    weights_to_save = {}

    # Diagnostic probes
    if label_info is not None:
        eid_to_pid, patient_labels, diag_cols = label_info

        if has_train and not using_weights:
            Y_train = build_label_matrix(train_ids, diag_cols, eid_to_pid, patient_labels)
        Y_val = build_label_matrix(val_ids, diag_cols, eid_to_pid, patient_labels)
        print(f"  Dx columns: {len(diag_cols)}", flush=True)

        for j, col in enumerate(diag_cols):
            y_va = Y_val[:, j]
            va_mask = ~np.isnan(y_va)

            if using_weights:
                coef_key = f"coef_{col}"
                if coef_key not in weights:
                    continue
                auroc, fpr, tpr, scores = score_with_weights(
                    Z_val_scaled[va_mask], y_va[va_mask],
                    weights[coef_key], float(weights[f"intercept_{col}"]),
                )
            else:
                y_tr = Y_train[:, j]
                tr_mask = ~np.isnan(y_tr)
                auroc, fpr, tpr, scores, coef, intercept = fit_probe(
                    Z_train_scaled[tr_mask], y_tr[tr_mask],
                    Z_val_scaled[va_mask], y_va[va_mask],
                )
                if auroc is not None:
                    weights_to_save[f"coef_{col}"] = coef
                    weights_to_save[f"intercept_{col}"] = np.array(intercept)

            if auroc is None:
                print(f"    Skipping {col} (insufficient positives)")
                continue

            n_pos_va = int(y_va[va_mask].sum())
            n_pos_tr = int(Y_train[:, j][~np.isnan(Y_train[:, j])].sum()) if has_train and not using_weights else 0
            print(f"    {col}: AUROC={auroc:.3f}  (val_pos={n_pos_va})")

            # Expand scores to full val array for UMAP plotting
            full_scores = np.full(len(val_ids), np.nan)
            full_scores[va_mask] = scores

            results.append(dict(
                target=col, auroc=auroc,
                fpr=fpr.tolist(), tpr=tpr.tolist(),
                n_pos_train=n_pos_tr, n_pos_val=n_pos_va,
                scores_val=full_scores, y_val=y_va,
            ))

    # Death probe (needs h5 + death_mrn)
    if dead_mrns is not None and data is not None:
        # Build meta for h5-available studies
        h5_val_mask = [i for i, s in enumerate(val_ids) if s in data]
        if len(h5_val_mask) == len(val_ids):
            # All val studies have h5 entries
            meta_val = build_meta(val_ids, data)
            death_Z_val_scaled = Z_val_scaled
            death_val_ids = val_ids
        else:
            # Subset to h5-available
            meta_val = build_meta([val_ids[i] for i in h5_val_mask], data)
            death_Z_val_scaled = Z_val_scaled[h5_val_mask]
            death_val_ids = [val_ids[i] for i in h5_val_mask]

        y_va, n_dead_va = build_death_vector(death_val_ids, meta_val, dead_mrns)

        if using_weights and "coef_Death" in weights:
            auroc, fpr, tpr, scores = score_with_weights(
                death_Z_val_scaled, y_va,
                weights["coef_Death"], float(weights["intercept_Death"]),
            )
        elif has_train and not using_weights:
            # Build death labels for train
            h5_tr_mask = [i for i, s in enumerate(train_ids) if s in data]
            meta_train = build_meta([train_ids[i] for i in h5_tr_mask], data)
            death_Z_train_scaled = Z_train_scaled[h5_tr_mask]
            y_tr, n_dead_tr = build_death_vector(
                [train_ids[i] for i in h5_tr_mask], meta_train, dead_mrns,
            )
            print(f"    Death: train_pos={n_dead_tr}, val_pos={n_dead_va}")
            auroc, fpr, tpr, scores, coef, intercept = fit_probe(
                death_Z_train_scaled, y_tr, death_Z_val_scaled, y_va,
            )
            if auroc is not None:
                weights_to_save["coef_Death"] = coef
                weights_to_save["intercept_Death"] = np.array(intercept)
        else:
            auroc = None

        if auroc is not None:
            print(f"    Death: AUROC={auroc:.3f}")
            # Expand scores back to full val indices if we subsetted
            if len(h5_val_mask) == len(val_ids):
                full_scores = scores
                full_y = y_va
            else:
                full_scores = np.full(len(val_ids), np.nan)
                full_scores[h5_val_mask] = scores
                full_y = np.full(len(val_ids), np.nan)
                full_y[h5_val_mask] = y_va

            results.append(dict(
                target="Death", auroc=auroc,
                fpr=fpr.tolist(), tpr=tpr.tolist(),
                n_pos_train=n_dead_tr if (has_train and not using_weights) else 0,
                n_pos_val=n_dead_va,
                scores_val=full_scores, y_val=full_y,
            ))
    elif dead_mrns is not None and data is None:
        print("    Skipping death probe (requires --h5_dir)", flush=True)

    
    # LVEF probe (Ridge regression)
    if lvef_map is not None:
        y_va_lvef = build_lvef_vector(val_ids, lvef_map)
        va_mask_lvef = ~np.isnan(y_va_lvef)

        if using_weights and "coef_LVEF" in weights:
            mae, preds = score_lvef_with_weights(
                Z_val_scaled[va_mask_lvef], y_va_lvef[va_mask_lvef],
                weights["coef_LVEF"], float(weights["intercept_LVEF"]),
            )
        elif has_train and not using_weights:
            y_tr_lvef = build_lvef_vector(train_ids, lvef_map)
            tr_mask_lvef = ~np.isnan(y_tr_lvef)
            mae, preds, coef, intercept = fit_lvef_probe(
                Z_train_scaled[tr_mask_lvef], y_tr_lvef[tr_mask_lvef],
                Z_val_scaled[va_mask_lvef], y_va_lvef[va_mask_lvef],
            )
            if mae is not None:
                weights_to_save["coef_LVEF"] = coef
                weights_to_save["intercept_LVEF"] = np.array(intercept)
        else:
            mae = None

        if mae is not None:
            print(f"    LVEF: MAE={mae:.2f}  (val_n={int(va_mask_lvef.sum())})")
            full_preds = np.full(len(val_ids), np.nan)
            full_preds[va_mask_lvef] = preds
            lvef_result = dict(target="LVEF", mae=mae, n_val=int(va_mask_lvef.sum()),
                               preds_val=full_preds, y_val=y_va_lvef)
        else:
            lvef_result = None
    else:
        lvef_result = None
    # Save results
    # Fit UMAP if anything to plot
    coords = None
    if results or lvef_result is not None:
        print("  Fitting UMAP for probe visualizations...", flush=True)
        coords = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42).fit_transform(Z_val)
        umap_dir = out / "umap_probes"
        umap_dir.mkdir(exist_ok=True)

    if results:
        csv_df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("fpr", "tpr", "scores_val", "y_val")}
        for r in results
        ])
        csv_df.sort_values("auroc", ascending=False).to_csv(out / "results.csv", index=False)
        print(f"  Saved {out / 'results.csv'}")
        plot_auroc_bar(results, out / "auroc_bar.png")
        plot_roc_curves(results, out / "roc_curves.png")

        for r in results:
            scores = r["scores_val"]
            y = r["y_val"]
            if scores is None or y is None:
                continue
            valid = ~np.isnan(scores) & ~np.isnan(y)
            pos = valid & (y == 1)
            if pos.sum() == 0:
                continue
            name = r["target"].lower().replace(" ", "_")
            plot_probe_umap(coords, y, scores, r["target"], r["auroc"],
                            umap_dir / f"umap_probe_{name}.png")

    # Save probe weights (only when fitting, not loading)
    if not using_weights and weights_to_save:
        weights_to_save["scaler_mean"] = scaler.mean_
        weights_to_save["scaler_scale"] = scaler.scale_
        np.savez_compressed(out / "probe_weights.npz", **weights_to_save)
        print(f"  Saved {out / 'probe_weights.npz'}")

    if lvef_result is not None:
        pd.DataFrame([{"target": "LVEF", "mae": lvef_result["mae"],
                       "n_val": lvef_result["n_val"]}]).to_csv(
            out / "results_lvef.csv", index=False)
        print(f"  Saved {out / 'results_lvef.csv'}")
        plot_lvef_umap(coords, lvef_result["y_val"], lvef_result["preds_val"],
                       lvef_result["mae"],
                       umap_dir / "umap_lvef_predicted.png",
                       umap_dir / "umap_lvef_error.png")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Supervised eval for report/study embeddings")
    p.add_argument("--report_embeddings",     default=None, help=".npz (new format)")
    p.add_argument("--study_embeddings",      default=None, help=".npz (new or old format)")
    p.add_argument("--h5_dir",                default=None, help="Line embeddings HDF5 dir (for death probe)")
    p.add_argument("--labels",                default=None, help="Fyler diagnostic labels CSV")
    p.add_argument("--death_mrn",             default=None, help="Death MRN CSV")
    p.add_argument("--report_probe_weights",  default=None, help="Pre-fit probe_weights.npz for report side")
    p.add_argument("--study_probe_weights",   default=None, help="Pre-fit probe_weights.npz for study side")
    p.add_argument("--lvef_labels",           default=None, help="CSV with Event.ID.Number + LVEF column")
    p.add_argument("--output_dir",            required=True)
    args = p.parse_args()

    assert args.report_embeddings or args.study_embeddings, \
        "At least one of --report_embeddings / --study_embeddings required"

    out = Path(args.output_dir)

    # Load shared resources
    data = preload_all(args.h5_dir) if args.h5_dir else None
    label_info = load_label_info(args.labels) if args.labels else None
    dead_mrns = load_death_mrns(args.death_mrn) if args.death_mrn else None
    lvef_map = load_lvef_labels(args.lvef_labels) if args.lvef_labels else None

    if args.death_mrn and not args.h5_dir:
        print("Warning: --death_mrn requires --h5_dir for MRN mapping, death probe will be skipped", flush=True)

    report_results, study_results = [], []

    if args.report_embeddings:
        print("\n=== Report side ===", flush=True)
        Z_train, train_ids, Z_val, val_ids = load_embeddings(args.report_embeddings)
        report_results = eval_side(
            Z_train, train_ids, Z_val, val_ids, out / "report",
            label_info=label_info, dead_mrns=dead_mrns, data=data,
            probe_weights_path=args.report_probe_weights,lvef_map=lvef_map,
        )

    if args.study_embeddings:
        print("\n=== Study side ===", flush=True)
        Z_train, train_ids, Z_val, val_ids = load_embeddings(args.study_embeddings)
        study_results = eval_side(
            Z_train, train_ids, Z_val, val_ids, out / "study",
            label_info=label_info, dead_mrns=dead_mrns, data=data,
            probe_weights_path=args.study_probe_weights,lvef_map=lvef_map,
        )

    # Comparison (only when both sides have results)
    if report_results and study_results:
        print("\n=== Comparison ===", flush=True)
        plot_comparison(report_results, study_results, out)

    print(f"\nAll done. Results in {out}")


if __name__ == "__main__":
    main()
