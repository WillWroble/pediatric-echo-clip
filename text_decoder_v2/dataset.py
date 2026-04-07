"""Dataset for line-level decoder. Centroid in, centroid out."""

import re
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


def load_study_embeddings(npz_path):
    """Load standard npz → (train_dict, val_dict, train_ids, val_ids)."""
    npz = np.load(npz_path)
    train_ids = npz["train_ids"].astype(str).tolist()
    val_ids = npz["val_ids"].astype(str).tolist()
    train_embs = {sid: emb for sid, emb in zip(train_ids, npz["train"].astype(np.float32))}
    val_embs = {sid: emb for sid, emb in zip(val_ids, npz["val"].astype(np.float32))}
    return train_embs, val_embs, train_ids, val_ids


def load_codebook(codebook_path):
    """Load centroids (K, 768) float32."""
    cb = np.load(codebook_path, allow_pickle=True)
    return cb["centroids"].astype(np.float32)


def load_ignore_patterns(path):
    """Load regex ignore patterns from file."""
    patterns = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(re.compile(line, re.IGNORECASE))
    print(f"Loaded {len(patterns)} ignore patterns", flush=True)
    return patterns


def should_ignore(text, patterns):
    return any(p.search(text) for p in patterns)


def load_manifest(path):
    """Load manifest → set of study IDs."""
    with open(path) as f:
        return set(l.strip() for l in f if l.strip())


def preload_lines(h5_path):
    """Load study_findings.h5 → {study_id: [line_str, ...]}."""
    data = {}
    with h5py.File(h5_path, "r") as f:
        for sid in f:
            lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                     for x in f[sid][()]]
            if lines:
                data[sid] = lines
    print(f"Loaded {len(data):,} studies from {h5_path}", flush=True)
    return data


def build_text_to_cid(all_lines_path, ignore_patterns):
    """Build text → centroid_id from all_lines.npz, applying ignore filter."""
    data = np.load(all_lines_path, allow_pickle=True)
    texts = data["texts"].astype(str).tolist()
    labels = data["labels"].astype(int)
    mapping = {}
    n_ignored = 0
    for text, label in zip(texts, labels):
        if should_ignore(text, ignore_patterns):
            n_ignored += 1
            continue
        mapping[text] = int(label)
    print(f"  {len(mapping):,} lines mapped, {n_ignored:,} ignored", flush=True)
    return mapping


class LineDataset(Dataset):
    def __init__(self, study_ids, line_data, study_embs, text_to_cid,
                 centroids, max_lines=40):
        self.max_lines = max_lines
        self.text_to_cid = text_to_cid
        self.centroids = centroids
        self.study_ids = [s for s in study_ids
                          if s in line_data and s in study_embs]
        self.line_data = line_data
        self.study_embs = study_embs

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        lines = self.line_data[sid]
        emb = self.study_embs[sid]

        cids = []
        for line in lines:
            cid = self.text_to_cid.get(line)
            if cid is not None:
                cids.append(cid)
        cids = cids[:self.max_lines]

        if cids:
            cent_embs = np.stack([self.centroids[c] for c in cids])
        else:
            cent_embs = np.zeros((1, 768), dtype=np.float32)

        return torch.from_numpy(cent_embs), torch.tensor(cids, dtype=torch.long), torch.from_numpy(emb)

def collate(batch):
    lines_list, cids_list, emb_list = zip(*batch)
    lengths = torch.tensor([l.shape[0] for l in lines_list], dtype=torch.long)
    L_max = lengths.max().item()
    D = lines_list[0].shape[1]
    B = len(batch)

    lines = torch.zeros(B, L_max, D)
    cids = torch.full((B, L_max), -1, dtype=torch.long)
    for i, (l, c) in enumerate(zip(lines_list, cids_list)):
        lines[i, :l.shape[0]] = l
        cids[i, :c.shape[0]] = c

    study_embs = torch.stack(emb_list).unsqueeze(1)
    return lines, cids, study_embs, lengths
