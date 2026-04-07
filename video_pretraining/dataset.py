"""Chunked clip dataset for contrastive pretraining."""

import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

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

def load_clips(hdf5_dir, manifest_path=None):
    """Load clip embeddings from HDF5 files into RAM.

    Returns:
        embeddings: (N_videos, 16, 768) float16 array.
        int_ids:    (N_videos,) int array mapping each video to unique_ids.
        unique_ids: (N_studies,) string array of unique study IDs.
    """
    if manifest_path is not None:
        keep = set(np.loadtxt(manifest_path, dtype=str).tolist())
        print(f"Manifest: {len(keep)} studies")

    files = sorted(f for f in os.listdir(hdf5_dir) if f.endswith(".hdf5"))
    if manifest_path is not None:
        files = [f for f in files if f.replace("_trim_embed.hdf5", "") in keep]
    print(f"Loading {len(files):,} HDF5 files")

    all_embs, all_ids = [], []
    for i, fname in enumerate(files):
        study_id = fname.replace("_trim_embed.hdf5", "")
        with h5py.File(os.path.join(hdf5_dir, fname), "r") as f:
            clip_embs = find_embeddings(f)
        for emb in clip_embs:
            if emb.shape == (16, 768):
                all_embs.append(emb.astype(np.float16))
                all_ids.append(study_id)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)} files loaded")

    embeddings = np.stack(all_embs)
    study_ids  = np.array(all_ids)
    unique_ids, int_ids = np.unique(study_ids, return_inverse=True)
    print(f"Loaded {len(embeddings):,} videos from {len(unique_ids):,} studies")
    return embeddings, int_ids, unique_ids

class ClipDataset(Dataset):
    """One item = one video (16 clips, 768d)."""

    def __init__(self, embeddings, study_ids):
        self.embeddings = embeddings  # float16 in RAM
        self.study_ids = torch.tensor(study_ids, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.embeddings[idx].astype(np.float32)),
            self.study_ids[idx],
        )
