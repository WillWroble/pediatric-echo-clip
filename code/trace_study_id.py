"""Trace a study ID through the echo report pipeline.

Usage:
    python trace_study_id.py --study_id 2204739
    python trace_study_id.py --study_id 2204739 --pdf_root /path/to/EchoReports
"""

import argparse
import glob
import h5py
import pandas as pd
from pathlib import Path

DEFAULTS = dict(
    csv_path="/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echo_reports_v2.csv",
    h5_dir="/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings",
    pdf_root="/lab-share/Cardio-Mayourian-e2/Public/AItext/EchoReports",
)


def find_pdf(study_id, pdf_root):
    matches = glob.glob(f"{pdf_root}/**/{study_id}_*.pdf", recursive=True)
    if not matches:
        # try without leading zeros
        matches = glob.glob(f"{pdf_root}/**/{study_id.lstrip('0')}_*.pdf", recursive=True)
    return matches


def find_csv_row(study_id, csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    row = df[df["study_id"] == study_id]
    if row.empty:
        row = df[df["study_id"] == study_id.lstrip("0")]
    return row


def find_h5(study_id, h5_dir):
    for h5_path in sorted(Path(h5_dir).glob("chunk_*.h5")):
        with h5py.File(h5_path, "r") as f:
            if study_id in f:
                grp = f[study_id]
                attrs = dict(grp.attrs)
                fields = {k: grp[k].shape for k in grp.keys()}
                return h5_path.name, attrs, fields
            # try without leading zeros
            sid = study_id.lstrip("0")
            if sid in f:
                grp = f[sid]
                attrs = dict(grp.attrs)
                fields = {k: grp[k].shape for k in grp.keys()}
                return h5_path.name, attrs, fields
    return None, None, None


def main():
    p = argparse.ArgumentParser(description="Trace a study ID through the pipeline")
    p.add_argument("--study_id", required=True)
    p.add_argument("--csv_path", default=DEFAULTS["csv_path"])
    p.add_argument("--h5_dir", default=DEFAULTS["h5_dir"])
    p.add_argument("--pdf_root", default=DEFAULTS["pdf_root"])
    args = p.parse_args()

    sid = args.study_id
    print(f"=== Tracing study_id: {sid} ===\n")

    # 1. PDF
    print("--- PDF ---")
    matches = find_pdf(sid, args.pdf_root)
    if matches:
        for m in matches:
            print(m)
    else:
        print("Not found")
    print()

    # 2. CSV
    print("--- CSV ---")
    row = find_csv_row(sid, args.csv_path)
    if row.empty:
        print("Not found")
    else:
        for col in row.columns:
            val = row[col].values[0]
            if pd.isna(val) or val == "":
                continue
            print(f"  {col}: {val}")
    print()

    # 3. HDF5
    print("--- HDF5 ---")
    chunk, attrs, fields = find_h5(sid, args.h5_dir)
    if chunk is None:
        print("Not found")
    else:
        print(f"  chunk: {chunk}")
        print(f"  demographics:")
        for k, v in attrs.items():
            print(f"    {k}: {v}")
        print(f"  embeddings:")
        for k, shape in fields.items():
            print(f"    {k}: {shape[0]} lines x {shape[1]}d")


if __name__ == "__main__":
    main()
