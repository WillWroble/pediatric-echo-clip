"""Train LineDecoder with geometry-informed soft BCE targets.

Usage:
    python -u train.py \
        --h5_path /lab-share/.../line_tokenizer/data/study_findings.h5 \
        --embeddings /lab-share/.../embeddings_video.npz \
        --line_embeddings /lab-share/.../line_embeddings.npz \
        --codebook /lab-share/.../codebook.npz \
        --output_dir results/v1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import LineDecoder
from dataset import LineDecoderDataset, assign_lines_to_clusters


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, lam=0):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_samples = 0.0, 0

    for embs, targets in loader:
        embs = embs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(training):
            logits = model(embs)
            sparsity = torch.sigmoid(logits).mean()
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss = loss + lam * sparsity

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item() * embs.size(0)
        total_samples += embs.size(0)

    return total_loss / total_samples


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--h5_path", required=True, help="preprocess H5 (study→lines)")
    p.add_argument("--embeddings", required=True, help="study embeddings npz")
    p.add_argument("--line_embeddings", required=True, help="encode.py output npz")
    p.add_argument("--codebook", required=True, help="codebook.npz from cluster.py")
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--val_manifest", default=None)
    p.add_argument("--line_filters", default=None)

    # soft labels
    p.add_argument("--tau", type=float, default=0.05)

    # model
    p.add_argument("--hidden_dim", type=int, default=768)

    # training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
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
    npz = np.load(args.embeddings)
    train_arr = npz["train"].astype(np.float32)
    train_ids = [str(int(float(x))) for x in npz["train_ids"]]
    val_arr = npz["val"].astype(np.float32)
    val_ids = [str(int(float(x))) for x in npz["val_ids"]]
    train_embs = {sid: train_arr[i] for i, sid in enumerate(train_ids)}
    val_embs = {sid: val_arr[i] for i, sid in enumerate(val_ids)}

    if args.train_manifest:
        manifest = set(str(int(float(x)))
                       for x in Path(args.train_manifest).read_text().strip().splitlines())
        train_embs = {s: e for s, e in train_embs.items() if s in manifest}
        print(f"Train manifest filter: {len(train_embs):,}")
    if args.val_manifest:
        manifest = set(str(int(float(x)))
                       for x in Path(args.val_manifest).read_text().strip().splitlines())
        val_embs = {s: e for s, e in val_embs.items() if s in manifest}
        print(f"Val manifest filter: {len(val_embs):,}")

    print("Assigning lines to clusters...", flush=True)
    line_to_cluster = assign_lines_to_clusters(args.line_embeddings, args.codebook)

    # Load codebook to get vocab size
    cb = np.load(args.codebook, allow_pickle=True)
    vocab_size = len(cb["cluster_ids"])

    train_ds = LineDecoderDataset(args.h5_path, list(train_embs.keys()), train_embs,
                                  line_to_cluster, args.codebook, tau=args.tau,
                                  line_filters=args.line_filters)
    val_ds = LineDecoderDataset(args.h5_path, list(val_embs.keys()), val_embs,
                                line_to_cluster, args.codebook, tau=args.tau,
                                line_filters=args.line_filters)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4)

    # ---- model ----
    model = LineDecoder(hidden_dim=args.hidden_dim, vocab_size=vocab_size, dropout=args.dropout).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True, strict=False))
        print(f"Loaded checkpoint: {args.checkpoint}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    # ---- train ----
    log_path = Path(args.output_dir) / "log.jsonl"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer, scheduler, lam=args.lam)
        val_loss = run_epoch(model, val_loader, device, lam=args.lam)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = dict(
            epoch=epoch, train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6), lr=round(lr, 6), time=round(elapsed, 1),
        )
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"lr={lr:.1e}  {elapsed:.0f}s",
            flush=True,
        )

        torch.save(model.state_dict(), Path(args.output_dir) / "latest.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "best.pt")
            print(f"  -> new best: {val_loss:.6f}", flush=True)

    print(f"\nBest val loss: {best_val_loss:.6f}", flush=True)
    with open(Path(args.output_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
