"""Train LineEncoder with video-attended skip-gram BCE objective."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import LineEncoder, CrossAttentionPool
from dataset import SkipGramDataset, collate_fn, load_videos_by_study


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(encoder, attn_pool, loader, device, L, optimizer=None, scheduler=None):
    training = optimizer is not None
    encoder.train() if training else encoder.eval()
    attn_pool.train() if training else attn_pool.eval()
    total_loss, total_pairs = 0.0, 0

    for input_ids, attn_mask, videos, video_mask, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        videos = videos.to(device)
        video_mask = video_mask.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            line_embs = encoder(input_ids, attn_mask)            # (B*L, 768)
            B = videos.shape[0]
            line_embs = line_embs.view(B, L, -1)                  # (B, L, 768)
            attended = attn_pool(line_embs, videos, video_mask)   # (B, L, 768)
            logits = (line_embs * attended).sum(dim=-1)           # (B, L)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(attn_pool.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        total_pairs += labels.size(0)

    return total_loss / total_pairs


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--h5_path", required=True)
    p.add_argument("--video_embeddings", required=True, help="infonce_768_all.npz")
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)

    # sampling
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--subsample_t", type=float, default=1e-3)
    p.add_argument("--max_videos", type=int, default=128)
    p.add_argument("--line_filters", default=None)

    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default=None)

    # output
    p.add_argument("--output_dir", required=True)

    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ---- data ----
    videos_by_study = load_videos_by_study(args.video_embeddings)

    train_manifest = set(str(int(float(x)))
                         for x in Path(args.train_manifest).read_text().strip().splitlines())
    val_manifest = set(str(int(float(x)))
                       for x in Path(args.val_manifest).read_text().strip().splitlines())
    train_ids = [s for s in train_manifest if s in videos_by_study]
    val_ids = [s for s in val_manifest if s in videos_by_study]
    print(f"Train: {len(train_ids):,}  Val: {len(val_ids):,}", flush=True)

    train_ds = SkipGramDataset(args.h5_path, train_ids, videos_by_study,
                                K=args.K, M=args.M, subsample_t=args.subsample_t,
                                max_videos=args.max_videos, line_filters=args.line_filters)
    val_ds = SkipGramDataset(args.h5_path, val_ids, videos_by_study,
                              K=args.K, M=args.M, subsample_t=args.subsample_t,
                              max_videos=args.max_videos, line_filters=args.line_filters)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    # ---- model ----
    encoder = LineEncoder().to(device)
    attn_pool = CrossAttentionPool(dim=768).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        if "attn_pool" in ckpt:
            attn_pool.load_state_dict(ckpt["attn_pool"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    n_total = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in attn_pool.parameters())
    n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
              sum(p.numel() for p in attn_pool.parameters() if p.requires_grad)
    print(f"Parameters: {n_train:,} trainable / {n_total:,} total", flush=True)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(attn_pool.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    L = args.K + args.M
    log_path = Path(args.output_dir) / "log.jsonl"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(encoder, attn_pool, train_loader, device, L, optimizer, scheduler)
        val_loss = run_epoch(encoder, attn_pool, val_loader, device, L)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = dict(epoch=epoch, train_loss=round(train_loss, 6),
                   val_loss=round(val_loss, 6), lr=round(lr, 6), time=round(elapsed, 1))
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={lr:.1e}  {elapsed:.0f}s", flush=True)

        ckpt = {"encoder": encoder.state_dict(), "attn_pool": attn_pool.state_dict()}
        torch.save(ckpt, Path(args.output_dir) / "latest.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, Path(args.output_dir) / "best.pt")
            print(f"  -> new best: {val_loss:.6f}", flush=True)

    print(f"\nBest val loss: {best_val_loss:.6f}", flush=True)
    with open(Path(args.output_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
