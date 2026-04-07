"""Train LineEncoder with skip-gram BCE objective.

Usage:
    python -u train.py \
        --h5_path data/study_findings.h5 \
        --embeddings results/full_contrast_v6/eval/embeddings_video.npz \
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
from transformers import AutoTokenizer

from model import LineEncoder
from dataset import SkipGramDataset, collate_fn

MAX_LENGTH = 128


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_pairs = 0.0, 0

    for input_ids, attn_mask, study_embs, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        study_embs = study_embs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            line_embs = model(input_ids, attn_mask)
            logits = (line_embs * study_embs).sum(dim=-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        total_pairs += labels.size(0)

    return total_loss / total_pairs


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--h5_path", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--val_manifest", default=None)

    # sampling
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--subsample_t", type=float, default=1e-3)
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

    train_ds = SkipGramDataset(args.h5_path, list(train_embs.keys()), train_embs,
                                K=args.K, M=args.M, subsample_t=args.subsample_t,
                                line_filters=args.line_filters)
    val_ds = SkipGramDataset(args.h5_path, list(val_embs.keys()), val_embs,
                              K=args.K, M=args.M, subsample_t=args.subsample_t,
                              line_filters=args.line_filters)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    # ---- model ----
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    model = LineEncoder().to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint}")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_total:,} total, {n_train:,} trainable", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    # ---- train ----
    log_path = Path(args.output_dir) / "log.jsonl"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer, scheduler)
        val_loss = run_epoch(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = dict(
            epoch=epoch, train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4), lr=round(lr, 6), time=round(elapsed, 1),
        )
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={lr:.1e}  {elapsed:.0f}s",
            flush=True,
        )

        torch.save(model.state_dict(), Path(args.output_dir) / "latest.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "best.pt")
            print(f"  -> new best: {val_loss:.4f}", flush=True)

    print(f"\nBest val loss: {best_val_loss:.4f}", flush=True)
    with open(Path(args.output_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
