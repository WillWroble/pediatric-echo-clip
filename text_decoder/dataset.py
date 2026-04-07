import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def preload_tokens(h5_path, study_ids, max_seq_len=512):
    """Load all token sequences into memory from HDF5."""
    tokens = {}
    with h5py.File(h5_path, "r") as f:
        for sid in study_ids:
            if sid in f:
                tokens[sid] = f[sid][:max_seq_len].astype(np.int16)
    print(f"Loaded tokens for {len(tokens):,} / {len(study_ids):,} studies", flush=True)
    return tokens


class FindingsDataset(Dataset):
    def __init__(self, study_ids, tokens, embeddings):
        """
        study_ids:  list of study ID strings
        tokens:     dict {study_id: 1D int64 array} (from preload_tokens)
        embeddings: dict {study_id: 768d numpy array}
        """
        self.ids = [s for s in study_ids if s in tokens and s in embeddings]
        self.tokens = tokens
        self.embeddings = embeddings
        print(f"FindingsDataset: {len(self.ids):,} / {len(study_ids):,} studies")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        return torch.tensor(self.tokens[sid]), torch.tensor(self.embeddings[sid])


def collate_fn(batch):
    """
    Returns:
        input_tokens:  (B, T) — tokens[:-1]
        target_tokens: (B, T) — tokens[1:]
        pad_mask:      (B, T) — True at pad positions
        cond:          (B, 1, D) — study embeddings
    """
    tokens_list, emb_list = zip(*batch)

    # teacher forcing split before padding
    inputs = [t[:-1] for t in tokens_list]
    targets = [t[1:] for t in tokens_list]

    max_len = max(t.shape[0] for t in inputs)

    input_tokens = torch.zeros(len(inputs), max_len, dtype=torch.long)
    target_tokens = torch.zeros(len(targets), max_len, dtype=torch.long)
    pad_mask = torch.ones(len(inputs), max_len, dtype=torch.bool)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = inp.shape[0]
        input_tokens[i, :L] = inp
        target_tokens[i, :L] = tgt
        pad_mask[i, :L] = False

    cond = torch.stack(emb_list).unsqueeze(1)  # (B, 1, D)

    return input_tokens, target_tokens, pad_mask, cond


def load_embeddings(npz_path, split):
    npz = np.load(npz_path)
    embs = npz[split].astype(np.float32)
    ids = npz[f"{split}_ids"].astype(str).tolist()
    return ids, embs
