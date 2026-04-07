"""Dataset for LineDecoder: study embeddings → soft target vectors over line vocabulary."""

import re
import numpy as np
import h5py
import torch
from pathlib import Path
from torch.utils.data import Dataset


def load_ignore_patterns(path):
    patterns = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, re.IGNORECASE))
    return patterns


def assign_lines_to_clusters(line_embs_path, codebook_path, batch_size=8192):
    """Assign every encoded line to its nearest codebook centroid.

    Returns dict: line_text → cluster_id
    """
    cb = np.load(codebook_path, allow_pickle=True)
    centroids = cb["centroids"].astype(np.float32)
    cluster_ids = cb["cluster_ids"]
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    le = np.load(line_embs_path, allow_pickle=True)
    lines = le["lines"].astype(str).tolist()
    embs = le["embeddings"].astype(np.float32)

    mapping = {}
    for i in range(0, len(lines), batch_size):
        batch_embs = embs[i:i + batch_size]
        batch_norm = batch_embs / (np.linalg.norm(batch_embs, axis=1, keepdims=True) + 1e-8)
        sims = batch_norm @ centroids_norm.T
        idxs = sims.argmax(axis=1)
        for j, line in enumerate(lines[i:i + batch_size]):
            mapping[line] = int(cluster_ids[idxs[j]])

    print(f"Assigned {len(mapping):,} lines to {len(cluster_ids)} clusters", flush=True)
    return mapping


class LineDecoderDataset(Dataset):

    def __init__(self, h5_path, study_ids, study_embs, line_to_cluster,
                 codebook_path, tau=0.05, line_filters=None):
        """
        Args:
            h5_path:         preprocess H5 (study_id → line texts)
            study_ids:       list of study IDs
            study_embs:      dict {study_id: 768d numpy array}
            line_to_cluster: dict {line_text: cluster_id}
            codebook_path:   path to codebook.npz
            tau:             temperature for soft labels
            line_filters:    path to regex ignore file
        """
        self.tau = tau

        if line_filters:
            patterns = [re.compile(l.strip(), re.IGNORECASE) for l in open(line_filters)
                        if l.strip() and not l.startswith("#")]
        else:
            patterns = []

        def keep(line):
            return not any(p.search(line) for p in patterns)

        # Load codebook
        cb = np.load(codebook_path, allow_pickle=True)
        self.cluster_ids = cb["cluster_ids"]
        centroids = cb["centroids"].astype(np.float32)
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        self.vocab_size = len(self.cluster_ids)
        self.cid_to_idx = {int(c): i for i, c in enumerate(self.cluster_ids)}

        # Precompute centroid cosine similarity matrix
        self.sim_matrix = centroids_norm @ centroids_norm.T  # (V, V)
        print(f"Codebook: {self.vocab_size} clusters, sim matrix {self.sim_matrix.shape}", flush=True)

        # Load study → active cluster indices
        embs_set = set(study_ids)
        self.study_active = {}
        with h5py.File(h5_path, "r") as f:
            for sid_raw in f.keys():
                sid = str(int(float(sid_raw)))
                if sid not in embs_set:
                    continue
                lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                         for x in f[sid_raw][()]]
                if patterns:
                    lines = [l for l in lines if keep(l)]
                active = set()
                for l in lines:
                    if l in line_to_cluster:
                        cid = line_to_cluster[l]
                        if cid in self.cid_to_idx:
                            active.add(self.cid_to_idx[cid])
                if active:
                    self.study_active[sid] = np.array(sorted(active), dtype=np.int64)

        self.study_ids = [s for s in study_ids if s in self.study_active]
        self.study_embs = study_embs
        # Cluster frequency counts (how many studies each cluster appears in)
        self.cluster_freqs = np.zeros(self.vocab_size, dtype=np.float32)
        for active in self.study_active.values():
            self.cluster_freqs[active] += 1

        print(f"LineDecoderDataset: {len(self.study_ids):,} studies, "
              f"mean {np.mean([len(v) for v in self.study_active.values()]):.1f} active tokens/study",
              flush=True)

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        emb = self.study_embs[sid]
        active = self.study_active[sid]

        # Build soft target
        sims_to_active = self.sim_matrix[:, active]  # (V, n_active)
        #weights = 1.0 / np.sqrt(self.cluster_freqs[active] + 1)
        #soft = (sims_to_active * weights[None, :]).sum(axis=1).astype(np.float32)
        #incoming_weights = 1.0 / np.sqrt(self.cluster_freqs + 1)  # (V,)
        #soft = sims_to_active.sum(axis=1) * incoming_weights
        soft = np.exp((sims_to_active - 1.0) / self.tau).sum(axis=1).astype(np.float32)
        soft = np.clip(soft, 0.0, 1.0)
        soft[active] = 1.0

        return torch.from_numpy(emb), torch.from_numpy(soft)
