"""Train EchoFocus Mini — transformer over shaped video embeddings.

Usage:
    python -u train_echofocusmini.py \
        --embeddings spectral_gelu.npz \
        --labels /path/to/labels.csv \
        --train_frac 0.1 \
        --output_dir results/spectral_frac0.1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from model import EchoFocus


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EchoDataset(Dataset):
    """One item = one echo study. Optionally samples a fixed number of videos."""

    def __init__(self, study_ids, emb_by_study, labels, n_videos_sample=None):
        self.study_ids = study_ids
        self.emb = emb_by_study
        self.labels = labels
        self.n_sample = n_videos_sample

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        emb = self.emb[sid]                         # (N_videos, D) np array
        label = np.float32(self.labels[sid])

        if self.n_sample is not None:
            n = emb.shape[0]
            replace = n < self.n_sample
            sel = np.random.choice(n, self.n_sample, replace=replace)
            emb = emb[sel]

        return torch.from_numpy(emb), torch.tensor(label)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args):
    """Load embeddings + labels, group by study, split 80/10/10 by patient."""

    # Embeddings: (N_clips, D) with parallel study_ids
    print(f"Loading {args.embeddings} ...", flush=True)
    data = np.load(args.embeddings, allow_pickle=True)
    all_embs = data["embeddings"]
    all_sids = data["study_ids"].astype(str)
    input_dim = all_embs.shape[1]
    print(f"  {all_embs.shape[0]:,} clips, {input_dim}d", flush=True)

    # Group clips → videos per study
    emb_by_study = {}
    for emb, sid in zip(all_embs, all_sids):
        emb_by_study.setdefault(sid, []).append(emb)
    emb_by_study = {k: np.stack(v, dtype=np.float32) for k, v in emb_by_study.items()}
    print(f"  {len(emb_by_study):,} unique studies", flush=True)

    # Labels
    df = pd.read_csv(args.labels)
    df["sid"] = df["Event.ID.Number"].astype(int).astype(str)
    df = df[["sid", "MRN", args.label_col]].drop_duplicates(subset="sid")
    df = df.dropna(subset=[args.label_col])

    # Join
    available = sorted(set(emb_by_study) & set(df["sid"]))
    df = df[df["sid"].isin(available)].set_index("sid")
    labels = df[args.label_col].to_dict()
    mrns = df["MRN"].to_dict()
    print(f"  {len(available):,} studies with labels", flush=True)

    # Split by patient (MRN)
    rng = np.random.RandomState(args.seed)
    unique_mrns = sorted(set(mrns[s] for s in available))
    rng.shuffle(unique_mrns)
    n = len(unique_mrns)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)

    train_mrns = set(unique_mrns[:n_tr])
    val_mrns = set(unique_mrns[n_tr : n_tr + n_va])
    test_mrns = set(unique_mrns[n_tr + n_va :])

    train_ids = [s for s in available if mrns[s] in train_mrns]
    val_ids = [s for s in available if mrns[s] in val_mrns]
    test_ids = [s for s in available if mrns[s] in test_mrns]

    # Subsample training set
    if args.train_frac < 1.0:
        k = max(1, int(len(train_ids) * args.train_frac))
        train_ids = sorted(rng.choice(train_ids, k, replace=False))

    print(f"  split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}", flush=True)
    return emb_by_study, labels, train_ids, val_ids, test_ids, input_dim


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Returns MAE (0–100 scale) and R²."""
    model.eval()
    preds, trues = [], []
    for emb, label in loader:
        out = model(emb.to(device)).squeeze(-1)
        preds.append(out.cpu())
        trues.append(label)
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    
    mae = (preds - trues).abs().mean().item() * 100
    ss_res = ((preds - trues) ** 2).sum()
    ss_tot = ((trues - trues.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    
    return mae, r2

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train EchoFocus Mini")

    # data
    p.add_argument("--embeddings", required=True, help=".npz with embeddings + study_ids")
    p.add_argument("--labels", required=True, help="CSV with Event.ID.Number + label column")
    p.add_argument("--label_col", default="LVEF", help="Column to predict (default: LVEF)")
    p.add_argument("--train_frac", type=float, default=1.0, help="Fraction of training set")

    # model
    p.add_argument("--input_dim", type=int, default=None, help="Auto-detected if omitted")
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--ff_dim", type=int, default=None, help="Defaults to input_dim")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--n_videos_sample", type=int, default=12)

    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # output
    p.add_argument("--output_dir", required=True)

    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ---- data ----
    emb_by_study, labels, train_ids, val_ids, test_ids, input_dim = load_data(args)
    if args.input_dim is not None:
        input_dim = args.input_dim
    if args.ff_dim is None:
        args.ff_dim = input_dim

    train_ds = EchoDataset(train_ids, emb_by_study, labels, n_videos_sample=args.n_videos_sample)
    val_ds = EchoDataset(val_ids, emb_by_study, labels)      # all videos, variable length
    test_ds = EchoDataset(test_ids, emb_by_study, labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)

    # ---- model ----
    model = EchoFocus(
        input_dim=input_dim,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        n_targets=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    loss_fn = nn.MSELoss()

    # ---- train ----
    history = []
    best_val_mae = float("inf")
    best_epoch = -1
    best_path = Path(args.output_dir) / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        t0 = time.time()

        for emb, label in train_loader:
            emb, label = emb.to(device), label.to(device)
            out = model(emb).squeeze(-1)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * emb.shape[0]
            n_seen += emb.shape[0]

        train_mse = total_loss / n_seen
        val_mae, val_r2 = evaluate(model, val_loader, device)
        scheduler.step(val_mae)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = dict(epoch=epoch, train_mse=train_mse, val_mae=val_mae, lr=lr, time=round(elapsed, 1))
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_mse={train_mse:.6f}  val_mae={val_mae:.2f}  r2={val_r2:.3f}  "
            f"lr={lr:.1e}  {elapsed:.0f}s",
            flush=True,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    # ---- test ----
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_mae, test_r2 = evaluate(model, test_loader, device)
    print(f"\nTest MAE: {test_mae:.2f}  R²: {test_r2:.4f}  (best epoch {best_epoch})", flush=True)


    # ---- save results ----
    results = dict(
        test_mae=round(test_mae, 4),
        test_r2=round(test_r2, 4),
        best_epoch=best_epoch,
        best_val_mae=round(best_val_mae, 4),
        n_train=len(train_ids),
        n_val=len(val_ids),
        n_test=len(test_ids),
        n_params=n_params,
        history=history,
        config=vars(args),
    )
    results_path = Path(args.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results_path}", flush=True)


if __name__ == "__main__":
    main()
