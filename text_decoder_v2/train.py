"""Train line-level decoder with MSE loss. Study embedding prepended, centroid in/out."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import LineDecoder
from dataset import (load_study_embeddings, load_codebook, load_ignore_patterns,
                     load_manifest, preload_lines, build_text_to_cid,
                     LineDataset, collate)


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, device, discrete=False, optimizer=None, scheduler=None, mask_ratio=0.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_positions = 0.0, 0

    for lines, cids, study_embs, lengths in loader:
        lines = lines.to(device)
        study_embs = study_embs.to(device)
        lengths = lengths.to(device)
        cids = cids.to(device)
        B, L, D = lines.shape

        # prepend study embedding at position 0
        x = torch.cat([study_embs, lines], dim=1)  # (B, 1+L, D)

        # padding mask: study emb always valid, then per-centroid
        pad_lines = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device),
                              pad_lines], dim=1)

        
        # mask input centroids during training to force study embedding reliance
        if is_train and mask_ratio > 0:
            noise = torch.rand(B, L, device=device)
            input_mask = noise < mask_ratio
            input_mask = input_mask & ~pad_lines  # only mask valid positions
            x[:, 1:][input_mask] = 0.0  # zero out masked centroids, keep study emb
        
        pred = model(x, pad_mask)  # (B, 1+L, D)


        if discrete:
            logits = model.head(pred[:, :L])  # (B, L, 10K)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   cids.reshape(-1), ignore_index=-1)
            n_valid = (cids != -1).sum().item()
        else:
            pred_lines = pred[:, :L]
            loss_mask = ~pad_lines
            diff = (pred_lines - lines) ** 2
            n_valid = loss_mask.sum()
            loss = (diff * loss_mask.unsqueeze(-1)).sum() / (n_valid * D)
        
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        nv = n_valid if isinstance(n_valid, int) else n_valid.item()
        total_loss += loss.item() * nv
        total_positions += nv
    return total_loss / total_positions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--discrete", action="store_true")
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--h5_path", required=True)
    p.add_argument("--all_lines", required=True)
    p.add_argument("--codebook", required=True)
    p.add_argument("--ignore_file", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--val_manifest", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_lines", type=int, default=40)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading study embeddings...", flush=True)
    train_embs, val_embs, train_ids, val_ids = load_study_embeddings(args.embeddings)

    if args.train_manifest:
        allowed = load_manifest(args.train_manifest)
        train_ids = [s for s in train_ids if s in allowed]
        print(f"Train manifest: {len(train_ids):,} studies", flush=True)
    if args.val_manifest:
        allowed = load_manifest(args.val_manifest)
        val_ids = [s for s in val_ids if s in allowed]
        print(f"Val manifest: {len(val_ids):,} studies", flush=True)

    print("Loading line data...", flush=True)
    line_data = preload_lines(args.h5_path)

    print("Building text→centroid map...", flush=True)
    ignore_patterns = load_ignore_patterns(args.ignore_file)
    text_to_cid = build_text_to_cid(args.all_lines, ignore_patterns)
    centroids = load_codebook(args.codebook)

    train_set = LineDataset(train_ids, line_data, train_embs, text_to_cid, centroids, args.max_lines)
    val_set = LineDataset(val_ids, line_data, val_embs, text_to_cid, centroids, args.max_lines)
    print(f"Train: {len(train_set):,}  Val: {len(val_set):,}", flush=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=True)

    model = LineDecoder(
        codebook_path=args.codebook, d_model=768, n_heads=args.n_heads,
        ff_dim=args.ff_dim, max_lines=args.max_lines, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, total_steps)

    best_val = float("inf")
    log_path = out / "log.jsonl"

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, args.discrete, optimizer, scheduler, args.mask_ratio)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, args.discrete)
        elapsed = time.time() - t0

        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 "lr": scheduler.get_last_lr()[0], "time": elapsed}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={entry['lr']:.2e}  {elapsed:.0f}s", flush=True)

        torch.save(model.state_dict(), out / "latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out / "best.pt")
            print(f"  → new best val loss: {best_val:.6f}", flush=True)

    print(f"Done. Best val loss: {best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
