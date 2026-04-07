"""Merge chunk npz files into a single clip_embeddings.npz."""
import os
import sys
import numpy as np


def main():
    chunk_dir = sys.argv[1]
    out_path = sys.argv[2]

    chunks = sorted(f for f in os.listdir(chunk_dir) if f.startswith("clip_chunk_") and f.endswith(".npz") and "part" in f)
    print(f"Merging {len(chunks)} chunks...")

    all_embs = []
    all_ids = []

    for i, fname in enumerate(chunks):
        d = np.load(os.path.join(chunk_dir, fname))
        all_embs.append(d["embeddings"].astype(np.float16))
        all_ids.append(d["study_ids"])
        if (i + 1) % 20 == 0:
            print(f"  loaded {i+1}/{len(chunks)}")

    embeddings = np.concatenate(all_embs).astype(np.float16)
    study_ids = np.concatenate(all_ids)

    np.savez(out_path, embeddings=embeddings, study_ids=study_ids)
    print(f"Saved {embeddings.shape} from {len(np.unique(study_ids))} studies to {out_path}")

    for fname in chunks:
        os.remove(os.path.join(chunk_dir, fname))
    print("Cleaned up chunk files")


if __name__ == "__main__":
    main()
