import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ReportDecoder
from dataset import FindingsDataset, collate_fn, preload_tokens


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_tokens = 0.0, 0

    for input_tok, target_tok, pad_mask, cond in loader:
        input_tok = input_tok.to(device)
        target_tok = target_tok.to(device)
        pad_mask = pad_mask.to(device)
        cond = cond.to(device)

        logits = model(input_tok, cond, pad_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tok.reshape(-1),
            ignore_index=0,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        n_tokens = (~pad_mask).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for input_tok, target_tok, pad_mask, cond in loader:
        input_tok = input_tok.to(device)
        target_tok = target_tok.to(device)
        pad_mask = pad_mask.to(device)
        cond = cond.to(device)

        logits = model(input_tok, cond, pad_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tok.reshape(-1),
            ignore_index=0,
        )

        n_tokens = (~pad_mask).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--h5_path", required=True)
    p.add_argument("--embeddings", required=True, help=".npz with train/val/train_ids/val_ids")
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--val_manifest", default=None)

    # model
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=512)

    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
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
    train_ids = npz["train_ids"].astype(str).tolist()
    val_arr = npz["val"].astype(np.float32)
    val_ids = npz["val_ids"].astype(str).tolist()
    train_embs = {sid: train_arr[i] for i, sid in enumerate(train_ids)}
    val_embs = {sid: val_arr[i] for i, sid in enumerate(val_ids)}

    tokens = preload_tokens(args.h5_path, train_ids + val_ids, args.max_seq_len)
    if args.train_manifest:
        manifest_ids = set(Path(args.train_manifest).read_text().strip().split("\n"))
        train_embs = {s: e for s, e in train_embs.items() if s in manifest_ids}
        print(f"Train manifest filter: {len(train_embs):,}")
    if args.val_manifest:
        manifest_ids = set(Path(args.val_manifest).read_text().strip().split("\n"))
        val_embs = {s: e for s, e in val_embs.items() if s in manifest_ids}
        print(f"Val manifest filter: {len(val_embs):,}")
    train_ds = FindingsDataset(train_ids, tokens, train_embs)
    val_ds = FindingsDataset(val_ids, tokens, val_embs)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # ---- model ----
    model = ReportDecoder(
        n_layers=args.n_layers, ff_dim=args.ff_dim,
        n_heads=args.n_heads, dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    ).to(device)

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
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = val_epoch(model, val_loader, device)
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
