"""Compute trajectory metrics for one embedding set, append to shared CSV.

Usage:
    python -u analyze_trajectory.py \
        --h5_dir /lab-share/.../Line_Embeddings \
        --csv /lab-share/.../echo_reports_v2.csv \
        --eval_dir results/vicreg_v6/eval \
        --output results/trajectory_analysis.csv
"""

import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def parse_study_date(s):
    try:
        return datetime.strptime(s, "%m%d%y%H%M%S")
    except Exception:
        return None


def load_metadata(h5_dir, study_ids):
    sid_set = set(study_ids)
    meta = {}
    for fpath in sorted(Path(h5_dir).glob("chunk_*.h5")):
        with h5py.File(fpath, "r") as f:
            for sid in f.keys():
                if sid in sid_set:
                    attrs = f[sid].attrs
                    meta[sid] = {
                        "mrn": attrs.get("mrn", ""),
                        "study_date": attrs.get("study_date", ""),
                        "age": attrs.get("age", ""),
                        "gender": attrs.get("gender", ""),
                    }
    return meta


def build_pairs(study_ids, Z, meta):
    sid_to_idx = {s: i for i, s in enumerate(study_ids)}
    patient_studies = defaultdict(list)
    for sid in study_ids:
        m = meta.get(sid)
        if not m:
            continue
        dt = parse_study_date(m["study_date"])
        if m["mrn"] and dt:
            patient_studies[m["mrn"]].append((dt, sid))

    pairs = []
    for mrn, studies in patient_studies.items():
        studies.sort()
        for i in range(len(studies) - 1):
            sid_t, sid_t1 = studies[i][1], studies[i + 1][1]
            if sid_t in sid_to_idx and sid_t1 in sid_to_idx:
                pairs.append({
                    "sid_t": sid_t,
                    "sid_t1": sid_t1,
                    "delta": Z[sid_to_idx[sid_t1]] - Z[sid_to_idx[sid_t]],
                })
    return pairs


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

AGE_BINS = [
    ("neonate",    0,    0.08),
    ("infant",     0.08, 1),
    ("toddler",   1,    3),
    ("child",      3,    12),
    ("adolescent", 12,   18),
    ("adult",      18,   200),
]


def parse_age_years(age_str):
    if not age_str or not age_str.strip():
        return None
    s = age_str.strip().lower()
    if s.endswith("m"):
        try:
            return float(s.rstrip("m ")) / 12
        except ValueError:
            return None
    s = s.replace("years", "").replace("year", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def age_bin(age_str):
    y = parse_age_years(age_str)
    if y is None:
        return None
    for label, lo, hi in AGE_BINS:
        if lo <= y < hi:
            return label
    return None


def assign_groups(pairs, meta, diag_map):
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        sid = p["sid_t"]
        groups[("baseline", "all")].append(i)

        diag = diag_map.get(sid)
        if diag:
            groups[("diagnosis", diag)].append(i)

        m = meta.get(sid, {})
        g = (m.get("gender") or "").strip()
        if g:
            groups[("gender", g.lower())].append(i)

        ab = age_bin(m.get("age", ""))
        if ab:
            groups[("age", ab)].append(i)

    return groups


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(deltas):
    t = torch.from_numpy(np.stack(deltas)).float()
    n = t.shape[0]
    magnitudes = t.norm(dim=1)
    normed = F.normalize(t, dim=1)

    # Mean cosine: (||sum(normed)||^2 - n) / (n*(n-1))
    s = normed.sum(dim=0)
    sum_all = float(s @ s)
    mean_cos = (sum_all - n) / (n * (n - 1))

    # E[cos^2] via Gram matrix: ||normed.T @ normed||_F^2
    G = normed.T @ normed  # (d, d) — small
    sum_sq = float((G * G).sum())
    e_cos2 = (sum_sq - n) / (n * (n - 1))
    std_cos = max(0, e_cos2 - mean_cos ** 2) ** 0.5

    return {
        "cos_mean": round(mean_cos, 4),
        "cos_std": round(std_cos, 4),
        "magnitude": round(float(magnitudes.mean()), 4),
        "n_pairs": n,
    }


# ---------------------------------------------------------------------------
# CSV append
# ---------------------------------------------------------------------------

def append_row(output_path, row_dict):
    path = Path(output_path)
    df = pd.DataFrame([row_dict])
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset="model", keep="last")
        combined.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--csv", required=True, help="echo_reports_v2.csv")
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--output", default="results/trajectory_analysis.csv")
    p.add_argument("--min_count", type=int, default=50)
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    model_name = eval_dir.parts[-2]  # results/{model}/eval
    print(f"Model: {model_name}", flush=True)

    data = np.load(eval_dir / "embeddings.npz", allow_pickle=True)
    Z = data["val"]
    study_ids = list(data["val_ids"])
    print(f"{len(study_ids):,} val studies, {Z.shape[1]}d", flush=True)

    # Diagnosis map
    df = pd.read_csv(args.csv, dtype=str)
    df = df[["study_id", "cardiac_history"]].drop_duplicates(subset="study_id")
    diag_map = {}
    for _, row in df.iterrows():
        text = str(row.get("cardiac_history", "")).strip().lower()
        if text == "nan":
            continue
        if text:
            diag_map[row["study_id"]] = text

    # Build pairs + groups
    meta = load_metadata(args.h5_dir, study_ids)
    pairs = build_pairs(study_ids, Z, meta)
    print(f"{len(pairs):,} trajectory pairs", flush=True)

    groups = assign_groups(pairs, meta, diag_map)
    groups = {
        k: v for k, v in groups.items()
        if k[0] == "baseline" or len(v) >= args.min_count
    }
    print(f"{len(groups)} groups after min_count={args.min_count}", flush=True)

    # Flatten into one row
    row = {"model": model_name}
    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (0 if k[0] == "baseline" else 1, k[0], -len(groups[k])),
    )
    for group_type, group_name in sorted_keys:
        deltas = [pairs[i]["delta"] for i in groups[(group_type, group_name)]]
        if len(deltas) < 2:
            continue
        m = compute_metrics(deltas)
        tag = f"{group_type}:{group_name}"
        for metric, val in m.items():
            row[f"{tag}__{metric}"] = val

    append_row(args.output, row)
    print(f"Appended {model_name} to {args.output}")


if __name__ == "__main__":
    main()
