"""Embed report text fields line-by-line with frozen ClinicalBERT.

Each text field is split on newlines. Each non-empty line gets a [CLS]
embedding (768d). Output is one HDF5 per chunk:

    /{study_id}/
        summary:         (N_lines, 768) float16
        study_findings:  (N_lines, 768) float16
        ...
        attrs: mrn, study_date, age, gender, weight_kg, height_cm, bsa, bmi

Usage:
    python embed_lines.py $SLURM_ARRAY_TASK_ID
"""

import sys
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

BASE = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip")
CHUNK_DIR = BASE / "echo_reports_chunks_v3"
MRN_DIR = BASE / "MRN_Chunks"
OUTPUT_DIR = BASE / "Line_Embeddings"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
BATCH_SIZE = 64
MAX_LENGTH = 128

TEXT_FIELDS = ["summary", "study_findings", "history", "measurements",
               "cardiac_history", "reason_for_exam"]

ATTR_FIELDS = ["mrn", "study_date", "dob", "age", "gender",
               "weight_kg", "height_cm", "bsa", "bmi"]


@torch.no_grad()
def embed_lines(lines, tokenizer, model, device):
    """Embed a list of strings, return (N, 768) float16 numpy array."""
    if not lines:
        return np.empty((0, 768), dtype=np.float16)

    all_cls = []
    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i : i + BATCH_SIZE]
        tokens = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(device)
        out = model(**tokens)
        cls = out.last_hidden_state[:, 0, :]
        all_cls.append(cls.cpu().half().numpy())

    return np.concatenate(all_cls, axis=0)


def split_lines(text):
    """Split text into non-empty stripped lines."""
    if not isinstance(text, str) or not text.strip():
        return []
    return [l.strip() for l in text.split("\n") if l.strip()]


def main():
    task_id = int(sys.argv[1])
    chunk_path = CHUNK_DIR / f"chunk_{task_id:04d}.csv"

    if not chunk_path.exists():
        print(f"Task {task_id}: {chunk_path} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task {task_id}: loading model on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    df = pd.read_csv(chunk_path, dtype=str).fillna("")

    # Derive study_date from filename
    df["study_date"] = df["file"].str.extract(r"_(\d+)\.pdf$")[0].fillna("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"chunk_{task_id:04d}.h5"
    n_studies = 0

    with h5py.File(out_path, "w") as f:
        for _, row in df.iterrows():
            sid = row.get("study_id", "")
            if not sid:
                continue


            if sid in f:
                continue
            grp = f.create_group(sid)

            # Store metadata as attributes
            for attr in ATTR_FIELDS:
                grp.attrs[attr] = row.get(attr, "")

            # Embed and store each text field
            for field in TEXT_FIELDS:
                lines = split_lines(row.get(field, ""))
                if lines:
                    emb = embed_lines(lines, tokenizer, model, device)
                    grp.create_dataset(field, data=emb, compression="gzip")

            n_studies += 1

    print(f"Task {task_id}: saved {n_studies} studies to {out_path}", flush=True)


if __name__ == "__main__":
    main()
