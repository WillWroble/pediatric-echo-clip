"""Contrastive pretraining with set attention encoder.

Usage:
    python -u train.py \
        --chunk_dir /path/to/clip_chunks \
        --manifest /path/to/pretrain_ids.txt \
        --loss spectral \
        --output pretrained_spectral.pt
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SetAttentionEncoder
from dataset import load_clips, ClipDataset


def spectral_contrastive_loss(z, study_ids):
    """Spectral contrastive loss (HaoChen et al., 2021)."""
    sim = z @ z.T
    pos_mask = (study_ids.unsqueeze(0) == study_ids.unsqueeze(1))
    pos_mask.fill_diagonal_(False)

    n_pos = pos_mask.sum()
    if n_pos == 0:
        return (sim ** 2).mean(), 0

    attraction = (sim * pos_mask).sum() / n_pos
    mask_no_diag = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    repulsion = (sim[mask_no_diag] ** 2).mean()

    return -2 * attraction + repulsion, n_pos.item()


def infonce_loss(z, study_ids, temperature=0.07):
    """InfoNCE / NT-Xent loss."""
    sim = z @ z.T / temperature
    pos_mask = (study_ids.unsqueeze(0) == study_ids.unsqueeze(1))
    pos_mask.fill_diagonal_(False)

    n_pos = pos_mask.sum()
    if n_pos == 0:
        return torch.tensor(0.0, device=z.device), 0

    sim.fill_diagonal_(float("-inf"))
    log_denom = torch.logsumexp(sim, dim=1)

    pos_sim = torch.where(pos_mask, sim, torch.zeros_like(sim))
    pos_per_anchor = pos_mask.sum(dim=1)
    has_pos = pos_per_anchor > 0
    loss_per_anchor = -(pos_sim.sum(dim=1)[has_pos] / pos_per_anchor[has_pos]) + log_denom[has_pos]

    return loss_per_anchor.mean(), n_pos.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_dir", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--output", default="pretrained.pt")
    p.add_argument("--input_dim", type=int, default=768)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["spectral", "infonce"], default="spectral")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    embeddings, int_ids, unique_ids = load_clips(args.hdf5_dir, args.manifest)
    dataset = ClipDataset(embeddings, int_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    model = SetAttentionEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"Loss: {args.loss}, batch_size: {args.batch_size}, lr: {args.lr}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = spectral_contrastive_loss if args.loss == "spectral" else infonce_loss

    for epoch in range(args.epochs):
        total_loss = 0
        total_pos = 0
        n_batches = 0

        for embs, sids in loader:
            embs, sids = embs.to(args.device), sids.to(args.device)
            z = model(embs)

            if args.loss == "spectral":
                loss, n_pos = loss_fn(z, sids)
            else:
                loss, n_pos = loss_fn(z, sids, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pos += n_pos
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_pos = total_pos / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  avg_pos={avg_pos:.0f}")

    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(args),
    }, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
