"""3D interactive UMAP visualization for report encoder embeddings.

Generates a single HTML file with a dropdown panel to switch between
color schemes (age, gender, study date, diagnoses, death, etc.).

Usage:
    python -u eval_v2.py \
        --embeddings results/v1/embeddings.npz \
        --h5_dir /lab-share/.../Line_Embeddings \
        --train_manifest manifests/train.txt \
        --val_manifest manifests/val.txt \
        --output_dir results/v1/eval_v2

Note: With 280K+ val points, the output HTML can be large (100MB+).
      Use --subsample N to cap the number of points if needed.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from umap import UMAP
import plotly.graph_objects as go

from report_dataset import preload_all, parse_study_date


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

def build_meta(val_ids, data):
    meta = []
    for sid in val_ids:
        lines, demos, mrn, sd_str = data[sid]
        dt = parse_study_date(sd_str)
        frac_year = (dt.year + (dt.timetuple().tm_yday - 1) / 365.25) if dt else np.nan
        meta.append(dict(
            study_id=sid, mrn=mrn,
            age=demos[0], gender=demos[1],
            weight_kg=demos[2], bsa=demos[4],
            n_lines=lines.shape[0],
            study_date=dt, frac_year=frac_year,
        ))
    return meta


def build_hover(meta):
    texts = []
    for m in meta:
        age_str = f"{m['age']:.1f}y" if not np.isnan(m["age"]) else "?"
        gender_str = {1.0: "M", 0.0: "F"}.get(m["gender"], "?")
        date_str = m["study_date"].strftime("%Y-%m") if m["study_date"] else "?"
        texts.append(
            f"Study: {m['study_id']}<br>"
            f"Age: {age_str}  Gender: {gender_str}<br>"
            f"Date: {date_str}  Lines: {m['n_lines']}"
        )
    return texts


# ---------------------------------------------------------------------------
# Color option builders
# ---------------------------------------------------------------------------

def continuous_option(name, values, colorscale, vmin=None, vmax=None):
    """Returns a plotly restyle button for a continuous color array."""
    valid = ~np.isnan(values)
    cmin = vmin if vmin is not None else float(np.nanpercentile(values, 2))
    cmax = vmax if vmax is not None else float(np.nanpercentile(values, 98))
    return dict(
        label=name,
        method="restyle",
        args=[{
            "marker.color": [values.tolist()],
            "marker.colorscale": [colorscale],
            "marker.cmin": [cmin],
            "marker.cmax": [cmax],
            "marker.showscale": [True],
        }],
    )


def categorical_option(name, color_list):
    """Returns a plotly restyle button for a pre-mapped list of color strings."""
    return dict(
        label=name,
        method="restyle",
        args=[{
            "marker.color": [color_list],
            "marker.colorscale": [None],
            "marker.cmin": [None],
            "marker.cmax": [None],
            "marker.showscale": [False],
        }],
    )


def gender_colors(meta):
    palette = {1.0: "#4878CF", 0.0: "#D65F5F"}
    return [palette.get(m["gender"], "#AAAAAA") for m in meta]


def binary_colors(vals, pos_color="#D65F5F", neg_color="#4878CF", na_color="#CCCCCC"):
    out = []
    for v in vals:
        if np.isnan(v):
            out.append(na_color)
        elif v == 1.0:
            out.append(pos_color)
        else:
            out.append(neg_color)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True, help=".npz from embed.py")
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--death_mrn", default=None)
    p.add_argument("--subsample", type=int, default=None,
                   help="Randomly subsample val set to N points for lighter HTML")
    p.add_argument("--n_neighbors", type=int, default=30)
    p.add_argument("--min_dist", type=float, default=0.3)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    npz = np.load(args.embeddings)
    Z_val   = npz["val"]
    val_ids = npz["val_ids"].astype(str).tolist()
    print(f"Val: {len(val_ids):,}", flush=True)

    # Load metadata
    data     = preload_all(args.h5_dir)
    val_mask = [i for i, s in enumerate(val_ids) if s in data]
    val_ids  = [val_ids[i] for i in val_mask]
    Z_val    = Z_val[val_mask]
    # Subsample
    if args.subsample and args.subsample < len(val_ids):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(val_ids), args.subsample, replace=False)
        idx.sort()
        val_ids = [val_ids[i] for i in idx]
        Z_val   = Z_val[idx]
        print(f"Subsampled to {len(val_ids):,}", flush=True)

    meta = build_meta(val_ids, data)
    hover = build_hover(meta)

    # 3D UMAP
    print("Fitting 3D UMAP...", flush=True)
    coords = UMAP(
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=42,
    ).fit_transform(Z_val)
    print("Done.", flush=True)

    # Build color options
    ages      = np.array([m["age"]       for m in meta])
    weights   = np.array([m["weight_kg"] for m in meta])
    bsas      = np.array([m["bsa"]       for m in meta])
    n_lines   = np.array([m["n_lines"]   for m in meta], dtype=float)
    frac_year = np.array([m["frac_year"] for m in meta])

    buttons = [
        continuous_option("Age",          ages,      "plasma",  vmin=0, vmax=25),
        categorical_option("Gender",      gender_colors(meta)),
        continuous_option("Weight (kg)",  weights,   "viridis"),
        continuous_option("BSA",          bsas,      "viridis"),
        continuous_option("Study Date",   frac_year, "viridis"),
        continuous_option("Report Lines", n_lines,   "inferno"),
    ]

    # Diagnostic labels
    diag_cols = []
    mrn_to_pid = {}
    patient_labels = None
    if args.labels:
        print("Loading diagnostic labels...", flush=True)
        label_df = pd.read_csv(args.labels, encoding="utf-8-sig")
        label_df["eid"] = label_df["eid"].astype(str)
        label_df["pid"] = label_df["pid"].astype(str)

        diag_cols = [c for c in [
            "Composite_Critical", "Composite_NonCritical",
            "VSD", "ASD", "TOF", "AVCD", "Coarct", "HLHS", "PDA",
            "TOF_ADJ", "AVCD_ADJ", "VSD_ADJ", "Coarct_ADJ",
        ] if c in label_df.columns]

        patient_labels = label_df.groupby("pid")[diag_cols].max()

        for _, row in label_df.iterrows():
            eid = row["eid"]
            if eid in data:
                _, _, mrn, _ = data[eid]
                if mrn:
                    mrn_to_pid[mrn] = row["pid"]

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
            buttons.append(categorical_option(f"{col} (n={n_pos})", binary_colors(vals)))

    # Death
    if args.death_mrn:
        print("Loading death data...", flush=True)
        death_df = pd.read_csv(args.death_mrn)
        death_df["mrn"] = death_df["mrn"].astype(str)
        dead_mrns = set(death_df["mrn"].str.lstrip("0"))

        patient_latest = {}
        for i, m in enumerate(meta):
            mrn_norm = m["mrn"].lstrip("0")
            if mrn_norm not in dead_mrns or m["study_date"] is None:
                continue
            if mrn_norm not in patient_latest or m["study_date"] > patient_latest[mrn_norm][0]:
                patient_latest[mrn_norm] = (m["study_date"], i)

        death_vals = np.zeros(len(val_ids))
        for _, idx in patient_latest.values():
            death_vals[idx] = 1.0
        n_dead = len(patient_latest)
        print(f"  {n_dead:,} deceased patients", flush=True)
        buttons.append(categorical_option(
            f"Death (n={n_dead})",
            binary_colors(death_vals, pos_color="#CC0000", neg_color="#DDDDDD"),
        ))

    # Build figure with first color option active
    first = buttons[0]["args"][0]
    fig = go.Figure(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=first["marker.color"][0],
            colorscale=first["marker.colorscale"][0],
            cmin=first["marker.cmin"][0],
            cmax=first["marker.cmax"][0],
            showscale=first["marker.showscale"][0],
            opacity=0.6,
            colorbar=dict(thickness=15, len=0.6),
        ),
        text=hover,
        hoverinfo="text",
    ))

    fig.update_layout(
        title="Report Encoder — 3D UMAP",
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            buttons=buttons,
            showactive=True,
            bgcolor="white",
            bordercolor="#AAAAAA",
        )],
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
    )

    html_path = out / "umap_3d.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Saved {html_path}", flush=True)

    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
