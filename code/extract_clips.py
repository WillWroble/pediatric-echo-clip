import os
import sys
import h5py
import numpy as np


def find_embeddings(group):
    embs = []
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset) and key == "emb":
            embs.append(item[:])
        elif isinstance(item, h5py.Group):
            embs.extend(find_embeddings(item))
    return embs


def main():
    hdf5_dir = sys.argv[1]
    out_dir = sys.argv[2]
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    files = sorted(f for f in os.listdir(hdf5_dir) if f.endswith(".hdf5"))

    # split files across tasks
    chunk = files[task_id::n_tasks]
    print(f"Task {task_id}/{n_tasks}: processing {len(chunk)} files")

    all_embs = []
    all_ids = []

    for i, fname in enumerate(chunk):
        study_id = fname.replace("_trim_embed.hdf5", "")
        path = os.path.join(hdf5_dir, fname)

        with h5py.File(path, "r") as f:
            clip_embs = find_embeddings(f)

        for emb in clip_embs:
            all_embs.append(emb.mean(axis=0))
            all_ids.append(study_id)

        if (i + 1) % 5000 == 0:
            print(f"  task {task_id}: {i + 1}/{len(chunk)} files, {len(all_embs)} clips")

    embeddings = np.stack(all_embs).astype(np.float32)
    study_ids = np.array(all_ids)

    out_path = os.path.join(out_dir, f"chunk_{task_id:04d}.npz")
    np.savez(out_path, embeddings=embeddings, study_ids=study_ids)
    print(f"Task {task_id}: saved {len(embeddings)} clips to {out_path}")


if __name__ == "__main__":
    main()
