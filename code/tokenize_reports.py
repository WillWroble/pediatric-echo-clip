"""Tokenize report text fields with ClinicalBERT WordPiece tokenizer.

Reads echo_reports_v3.csv, tokenizes each TEXT_FIELD, writes one HDF5 per field.
Each study is stored as a 1D int16 array of token IDs (variable length).

Output:
    Report_Tokens/
        summary.h5
        study_findings.h5
        history.h5
        measurements.h5
        cardiac_history.h5
        reason_for_exam.h5

Usage:
    python -u tokenize_reports.py
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from transformers import AutoTokenizer

BASE = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip")
CSV_PATH = BASE / "echo_reports_v3.csv"
OUTPUT_DIR = BASE / "Report_Tokens"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

TEXT_FIELDS = [
    "summary", "study_findings", "history",
    "measurements", "cardiac_history", "reason_for_exam",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Reading {CSV_PATH}...", flush=True)
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

    print(f"Loaded {len(df):,} rows", flush=True)

    for field in TEXT_FIELDS:
        out_path = OUTPUT_DIR / f"{field}.h5"
        n_stored = 0
        n_skipped = 0

        with h5py.File(out_path, "w") as f:
            for _, row in df.iterrows():
                sid = row.get("study_id", "")
                text = row.get(field, "").strip()
                if not sid or not text:
                    n_skipped += 1
                    continue
                if sid in f:
                    continue

                token_ids = tokenizer.encode(text)
                f.create_dataset(sid, data=np.array(token_ids, dtype=np.int16))
                n_stored += 1

        print(f"{field}: {n_stored:,} stored, {n_skipped:,} skipped → {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
