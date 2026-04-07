"""Generate downstream video embeddings from a pretrained model.

Usage:
    python -u embed.py \
        --checkpoint pretrained_spectral.pt \
        --hdf5_dir /path/to/clip chunks \
        --manifest /path/to/downstream_ids.txt \
        --output spectral_attention.npz
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import SetAttentionEncoder
from dataset import load_clips, ClipDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hdf5_dir", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--output", default="embeddings.npz")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    train_args = ckpt["args"]

    model = SetAttentionEncoder(
        input_dim=train_args["input_dim"],
        hidden_dim=train_args["hidden_dim"],
        n_heads=train_args["n_heads"],
    ).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    embeddings, int_ids, unique_ids = load_clips(args.hdf5_dir, args.manifest)
    dataset = ClipDataset(embeddings, int_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_embs = []
    all_idx = []
    offset = 0

    with torch.no_grad():
        for embs, sids in loader:
            h = model.encode(embs.to(args.device))
            all_embs.append(h.cpu().numpy())
            n = embs.shape[0]
            all_idx.append(np.arange(offset, offset + n))
            offset += n

    all_embs = np.concatenate(all_embs).astype(np.float32)

    # map int IDs back to string study IDs
    study_id_strings = unique_ids[int_ids]

    np.savez(args.output, embeddings=all_embs, study_ids=study_id_strings)
    print(f"Saved {all_embs.shape} to {args.output}")


if __name__ == "__main__":
    main()
