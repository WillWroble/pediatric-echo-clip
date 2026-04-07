"""Extract raw (16, 768) clip embeddings per video from HDF5 files.

SLURM array job — each task processes a chunk of files.
Flushes to disk every 50K videos to avoid OOM.
"""
import os
import sys
import h5py
import numpy as np


def find_embeddings(group):
    """Recursively find all 'emb' datasets in an HDF5 group."""
    embs = []
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset) and key == "emb":
            embs.append(item[:])
        elif isinstance(item, h5py.Group):
            embs.extend(find_embeddings(item))
    return embs


FLUSH_EVERY = 50_000


def main():
    hdf5_dir = sys.argv[1]
    out_dir = sys.argv[2]
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    files = sorted(f for f in os.listdir(hdf5_dir) if f.endswith(".hdf5"))
    chunk = files[task_id::n_tasks]
    print(f"Task {task_id}/{n_tasks}: processing {len(chunk)} files")

    buf_embs = []
    buf_ids = []
    part = 0
    total = 0
    skipped = 0

    def flush():
        nonlocal buf_embs, buf_ids, part, total
        if not buf_embs:
            return
        embs = np.stack(buf_embs).astype(np.float32)
        ids = np.array(buf_ids)
        out_path = os.path.join(out_dir, f"clip_chunk_{task_id:04d}_part{part:03d}.npz")
        np.savez(out_path, embeddings=embs, study_ids=ids)
        total += len(embs)
        print(f"  task {task_id}: flushed {len(embs)} videos to part {part} ({total} total)")
        buf_embs.clear()
        buf_ids.clear()
        part += 1

    for i, fname in enumerate(chunk):
        study_id = fname.replace("_trim_embed.hdf5", "")
        path = os.path.join(hdf5_dir, fname)

        with h5py.File(path, "r") as f:
            clip_embs = find_embeddings(f)

        for emb in clip_embs:
            if emb.shape == (16, 768):
                buf_embs.append(emb)
                buf_ids.append(study_id)
            else:
                skipped += 1

        if len(buf_embs) >= FLUSH_EVERY:
            flush()

    flush()
    print(f"Task {task_id}: done. {total} videos, {skipped} skipped, {part} parts")


if __name__ == "__main__":
    main()
