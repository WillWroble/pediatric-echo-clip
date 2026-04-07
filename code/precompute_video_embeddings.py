"""Pre-compute mean-pooled study-level video embeddings from PanEcho HDF5 files."""

import numpy as np
import h5py
import os
from pathlib import Path

EMBED_DIR = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Embeddings/Embeddings/Internal")
OVERLAP_IDS = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/analysis/overlap_study_ids.txt")
OUT_PATH = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Video_Embeddings/study_embeddings.npz")


def load_study_embedding(path):
    """Mean-pool frames per sub-video, then mean-pool sub-videos per study."""
    vecs = []
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/emb"):
            vecs.append(obj[:].mean(axis=0))  # (16, 768) -> (768,)
    with h5py.File(path, "r") as f:
        f.visititems(collect)
    if not vecs:
        return None
    return np.stack(vecs).mean(axis=0)  # (768,)


def main():
    study_ids = [line.strip() for line in open(OVERLAP_IDS) if line.strip()]
    print(f"Processing {len(study_ids)} studies")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    valid_ids = []
    embeddings = []
    failed = 0

    for i, sid in enumerate(study_ids):
        hdf5_path = EMBED_DIR / f"{sid}_trim_embed.hdf5"
        if not hdf5_path.exists():
            failed += 1
            continue
        emb = load_study_embedding(str(hdf5_path))
        if emb is not None:
            valid_ids.append(sid)
            embeddings.append(emb)
        else:
            failed += 1

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(study_ids)} done, {len(valid_ids)} valid, {failed} failed")

    embeddings = np.stack(embeddings).astype(np.float32)
    print(f"Final: {embeddings.shape} ({failed} failed)")

    np.savez(OUT_PATH, study_ids=np.array(valid_ids), embeddings=embeddings)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
