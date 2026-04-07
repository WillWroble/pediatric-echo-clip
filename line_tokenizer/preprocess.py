"""Extract per-line text from Line_Embeddings_v2 into flat per-field H5 files.

Output: one H5 per field, each containing study_id groups with line text arrays.

Usage:
    python -u preprocess.py \
        --h5_dir /lab-share/.../Line_Embeddings_v2 \
        --output_dir /lab-share/.../line_tokenizer/data
"""

import argparse
from pathlib import Path

import h5py


TEXT_FIELDS = [
    "summary", "study_findings", "history",
    "measurements", "cardiac_history", "reason_for_exam",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    chunks = sorted(Path(args.h5_dir).glob("chunk_*.h5"))
    print(f"Found {len(chunks)} chunks", flush=True)

    dt = h5py.string_dtype(encoding="utf-8")
    writers = {
        field: h5py.File(Path(args.output_dir) / f"{field}.h5", "w")
        for field in TEXT_FIELDS
    }

    n_studies = 0
    for i, chunk_path in enumerate(chunks):
        with h5py.File(chunk_path, "r") as f:
            for sid in f.keys():
                for field in TEXT_FIELDS:
                    key = f"{field}_text"
                    if key in f[sid]:
                        lines = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f[sid][key][()]]
                        if lines and sid not in writers[field]:
                            writers[field].create_dataset(sid, data=lines, dtype=dt)
                n_studies += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(chunks)} chunks, {n_studies:,} studies", flush=True)

    for field, fh in writers.items():
        print(f"{field}: {len(fh.keys()):,} studies", flush=True)
        fh.close()

    print(f"Done. {n_studies:,} total studies.", flush=True)


if __name__ == "__main__":
    main()
