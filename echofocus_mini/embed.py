"""Export study-level embeddings from a trained EchoFocus Mini model.

Usage:
    python -u embed.py \
        --checkpoint results/infonce_768/best.pt \
        --embeddings infonce_768.npz \
        --output /lab-share/.../study_embeddings_v2.npz
"""

import argparse
import numpy as np
import torch
from model import EchoFocus


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", required=True, help=".npz with video embeddings + study_ids")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Load video embeddings, group by study
    data = np.load(args.embeddings)
    all_embs = data["embeddings"]
    all_sids = data["study_ids"].astype(str)
    input_dim = all_embs.shape[1]

    emb_by_study = {}
    for emb, sid in zip(all_embs, all_sids):
        emb_by_study.setdefault(sid, []).append(emb)
    emb_by_study = {k: np.stack(v, dtype=np.float32) for k, v in emb_by_study.items()}
    print(f"Loaded {len(emb_by_study):,} studies, {input_dim}d")

    # Load model
    model = EchoFocus(input_dim=input_dim, n_heads=8, ff_dim=input_dim, dropout=0.0, n_targets=1)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.to(args.device)
    model.eval()

    # Extract embeddings (pre-head: encoder → mean pool → layernorm)
    study_ids = sorted(emb_by_study.keys())
    embeddings = []
    for sid in study_ids:
        x = torch.from_numpy(emb_by_study[sid]).unsqueeze(0).to(args.device)
        h = model.encoder(x)
        h = h.mean(dim=1)
        h = model.norm(h)
        embeddings.append(h.squeeze(0).cpu().numpy())

    embeddings = np.stack(embeddings).astype(np.float32)
    np.savez(args.output, embeddings=embeddings, study_ids=np.array(study_ids))
    print(f"Saved {embeddings.shape} to {args.output}")


if __name__ == "__main__":
    main()
