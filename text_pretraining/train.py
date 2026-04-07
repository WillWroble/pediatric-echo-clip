
"""Train report encoder with VICReg, trajectory, and/or contrastive objectives.

Usage:
    # VICReg only
    python -u train.py --vicreg --h5_dir ... --train_manifest ... --output_dir results/v1

    # Trajectory only (from checkpoint)
    python -u train.py --trajectory --checkpoint results/v1/latest.pt ...

    # Contrastive alignment only
    python -u train.py --contrast --video_embeddings study_embs.npz --h5_dir ... --train_manifest ... --output_dir results/con

    # Joint (any combination)
    python -u train.py --vicreg --contrast --video_embeddings study_embs.npz --h5_dir ... --output_dir results/joint
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from report_encoder import ReportEncoder
from report_dataset import (
    preload_all, compute_demo_stats, build_trajectory_pairs,
    VICRegDataset, vicreg_collate, TrajectoryDataset, trajectory_collate,
    ContrastDataset, contrast_collate, load_video_embeddings,
    VideoContrastDataset, video_contrast_collate, load_video_embeddings_by_study,
    PatientBatchSampler,
)
# CKA independence test
def cka_temporal(z, t, D):
    B = z.shape[0]
    K = torch.exp(-torch.cdist(z, z).pow(2) / (2 * D))
    t_c = t.nan_to_num(t.nanmean()).unsqueeze(1)
    L = torch.exp(-torch.cdist(t_c, t_c).pow(2) / 50.0)
    H = torch.eye(B, device=z.device) - 1.0 / B
    KH = K @ H
    LH = L @ H
    hsic_kl = (KH @ LH).trace()
    hsic_kk = (KH @ KH).trace()
    hsic_ll = (LH @ LH).trace()
    return hsic_kl / (hsic_kk * hsic_ll).sqrt().clamp(min=1e-8)
# ---------------------------------------------------------------------------
# VICReg loss
# ---------------------------------------------------------------------------

def vicreg_loss(z1, z2, t, lam=25.0, mu=25.0, nu=1.0, tau=0.0):
    B, D = z1.shape

    inv = F.mse_loss(z1, z2)

    std1, std2 = z1.std(0), z2.std(0)
    var = (torch.relu(1.0 - std1).mean() + torch.relu(1.0 - std2).mean()) / 2

    z1c, z2c = z1 - z1.mean(0), z2 - z2.mean(0)
    cov1 = (z1c.T @ z1c) / (B - 1)
    cov2 = (z2c.T @ z2c) / (B - 1)
    off = ~torch.eye(D, device=z1.device, dtype=torch.bool)
    cov = (cov1[off].pow(2).sum() + cov2[off].pow(2).sum()) / (2 * D)
    # temporal independence (HSIC)
    z_avg = (z1 + z2) / 2
    
    t_c = t.nan_to_num(t.nanmean()).unsqueeze(1)                    # (B, 1)
    K = torch.exp(-torch.cdist(z_avg, z_avg).pow(2) / (2 * D))     # σ_z² = D
    L = torch.exp(-torch.cdist(t_c, t_c).pow(2) / 50.0)            # σ_t² = 25 (~5 yrs)
    H = torch.eye(B, device=z1.device) - 1.0 / B
    temporal = (K @ H @ L @ H).trace() / (B - 1) ** 2
    temporal = temporal*1000.0
    
    #temporal = cka_temporal(z_avg, t, D)

    return lam * inv + mu * var + nu * cov + tau * temporal, {
        "inv": inv.item(), "var": var.item(), "cov": cov.item(), "temporal": temporal.item(),
    }

# ---------------------------------------------------------------------------
# Contrastive alignment loss (report ↔ video)
# ---------------------------------------------------------------------------

def contrast_loss(z_report, z_video, temperature=0.07):
    """InfoNCE between report and vision embeddings."""
    z_video = F.normalize(z_video, dim=-1)
    sim = z_report @ z_video.T / temperature
    labels = torch.arange(len(z_report), device=z_report.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

# ---------------------------------------------------------------------------
# Training step helpers
# ---------------------------------------------------------------------------

def vicreg_step(encoder, batch, device, lam, mu, nu, tau=0.0, bernoulli=False):
    if bernoulli:
        lines1, demos1, mask1, lines2, demos2, mask2, t = [x.to(device) for x in batch]
        z1 = encoder.vicreg(lines1, demos1, mask1)
        z2 = encoder.vicreg(lines2, demos2, mask2)
    else:
        lines1, demos1, lines2, demos2, t = [x.to(device) for x in batch]
        z1 = encoder.vicreg(lines1, demos1)
        z2 = encoder.vicreg(lines2, demos2)
    loss, components = vicreg_loss(z1, z2, t, lam, mu, nu, tau)
    return loss, lines1.shape[0], components


def trajectory_step(encoder, batch, device):
    lines_t, demos_t, mask_t, lines_t1, demos_t1, mask_t1 = [
        x.to(device) if torch.is_tensor(x) else x for x in batch
    ]
    # actual delta lives in encode space
    e_t = encoder.encode(lines_t, demos_t, mask_t)
    e_t1 = encoder.encode(lines_t1, demos_t1, mask_t1)
    delta_actual = F.normalize(e_t1 - e_t, dim=-1)

    # predicted direction from traj projection of e_t
    delta_pred = encoder.traj(lines_t, demos_t, mask_t)

    loss = (1 - F.cosine_similarity(delta_pred, delta_actual)).mean()
    return loss, lines_t.shape[0]

def contrast_step(encoder, batch, device, temperature=0.07, echofocus=None):
    lines, demos, mask, video = batch
    lines, demos, video = lines.to(device), demos.to(device), video.to(device)
    if mask is not None:
        mask = mask.to(device)
    z_report = encoder.contrast(lines, demos, mask)
    z_video = echofocus.contrast(video) if echofocus is not None else video
    loss = contrast_loss(z_report, z_video, temperature)
    return loss, lines.shape[0]
# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_trajectory(encoder, loader, device):
    encoder.eval()
    total, seen = 0.0, 0
    for batch in loader:
        lines_t, demos_t, mask_t, lines_t1, demos_t1, mask_t1 = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]
        e_t = encoder.encode(lines_t, demos_t, mask_t)
        e_t1 = encoder.encode(lines_t1, demos_t1, mask_t1)
        delta_actual = F.normalize(e_t1 - e_t, dim=-1)
        delta_pred = encoder.traj(lines_t, demos_t, mask_t)
        loss = (1 - F.cosine_similarity(delta_pred, delta_actual)).mean()
        total += loss.item() * lines_t.shape[0]
        seen += lines_t.shape[0]
    return total / seen if seen > 0 else float("inf")

@torch.no_grad()
def validate_contrast(encoder, loader, device, temperature=0.07, echofocus=None):
    encoder.eval()
    if echofocus is not None:
        echofocus.eval()
    total, seen = 0.0, 0
    for batch in loader:
        loss, bs = contrast_step(encoder, batch, device, temperature, echofocus)
        total += loss.item() * bs
        seen += bs
    return total / seen if seen > 0 else float("inf")
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train report encoder")

    # objectives
    p.add_argument("--vicreg", action="store_true")
    p.add_argument("--trajectory", action="store_true")
    p.add_argument("--traj_weight", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.0, help="Temporal decorrelation weight")
    p.add_argument("--contrast", action="store_true")
    p.add_argument("--contrast_weight", type=float, default=1.0)
    p.add_argument("--study_embeddings", default=None, help=".npz of frozen study-level video embeddings")
    p.add_argument("--video_embeddings", default=None, help=".npz of video-level embeddings for joint EchoFocus training")
    p.add_argument("--echofocus_checkpoint", default=None, help="Optional warm-start for EchoFocus")
    p.add_argument("--n_videos_sample", type=int, default=12)
    p.add_argument("--contrast_temperature", type=float, default=0.07)

    # data
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", default=None)
    p.add_argument("--n_sample", type=int, default=30, help="Lines per VICReg view")

    # model
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    p.add_argument("--freeze_trunk", action="store_true", help="Freeze attention/norm, train only fuse + projection heads")
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--demo_dropout", type=float, default=0.5)
    p.add_argument("--bernoulli_p", type=float, default=None, help="Bernoulli line dropout rate (replaces fixed n_sample)")

    # vicreg
    p.add_argument("--lam", type=float, default=25.0)
    p.add_argument("--mu", type=float, default=25.0)
    p.add_argument("--nu", type=float, default=1.0)

    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--patient_batch", action="store_true", help="Sample one study per patient per batch")
    p.add_argument("--batch_size_traj", type=int, default=1024)
    p.add_argument("--batch_size_contrast", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # output
    p.add_argument("--output_dir", required=True)

    args = p.parse_args()
    assert args.vicreg or args.trajectory or args.contrast, "Enable --vicreg and/or --trajectory and/or --contrast"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Objectives: vicreg={args.vicreg}  trajectory={args.trajectory}  contrast={args.contrast}")

    # ---- data ----
    data = preload_all(args.h5_dir)
    manifest_ids = [s for s in Path(args.train_manifest).read_text().strip().splitlines() if s in data]
    print(f"Train studies: {len(manifest_ids):,}")

    # ---- encoder ----
    demo_mean, demo_std = None, None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
        cfg = ckpt["config"]
        demo_mean, demo_std = ckpt["demo_mean"], ckpt["demo_std"]
        encoder = ReportEncoder(
            input_dim=768, n_heads=cfg.get("n_heads", args.n_heads),
            dropout=args.dropout, n_demo=6,
        ).to(device)
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    else:
        demo_mean, demo_std = compute_demo_stats(data)
        encoder = ReportEncoder(
            input_dim=768, n_heads=args.n_heads, dropout=args.dropout, n_demo=6,
        ).to(device)

    n_params = sum(pp.numel() for pp in encoder.parameters())
    print(f"Encoder: {n_params:,} params")

    
    if args.freeze_trunk:
        for name, param in encoder.named_parameters():
            if name.startswith(("query", "attention", "norm")):
                param.requires_grad = False
        n_frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
        n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"Frozen trunk: {n_frozen:,} params, trainable: {n_trainable:,} params")
    
    # ---- dataloaders ----
    vicreg_loader = None
    if args.vicreg:
        vicreg_ds = VICRegDataset(
            manifest_ids, data, demo_mean, demo_std,
            n_sample=args.n_sample, demo_dropout=args.demo_dropout,
            bernoulli_p=args.bernoulli_p,
        )
        if args.patient_batch:
            sampler = PatientBatchSampler(manifest_ids, data, batch_size=args.batch_size)
            vicreg_loader = DataLoader(
                vicreg_ds, batch_sampler=sampler, num_workers=0,
                collate_fn=vicreg_collate if args.bernoulli_p else None,
            )
        else:
            vicreg_loader = DataLoader(
                vicreg_ds, batch_size=args.batch_size,
                shuffle=True, drop_last=True, num_workers=0,
                collate_fn=vicreg_collate if args.bernoulli_p else None,
            )

    traj_loader, val_traj_loader = None, None
    if args.trajectory:
        pairs = build_trajectory_pairs(manifest_ids, data)
        print(f"Train trajectory pairs: {len(pairs):,}")
        traj_ds = TrajectoryDataset(pairs, data, demo_mean, demo_std)
        traj_loader = DataLoader(
            traj_ds, batch_size=args.batch_size_traj,
            shuffle=True, drop_last=True, num_workers=0,
            collate_fn=trajectory_collate,
        )
        if args.val_manifest:
            val_ids = [s for s in Path(args.val_manifest).read_text().strip().splitlines() if s in data]
            val_pairs = build_trajectory_pairs(val_ids, data)
            print(f"Val trajectory pairs: {len(val_pairs):,}")
            val_traj_ds = TrajectoryDataset(val_pairs, data, demo_mean, demo_std)
            val_traj_loader = DataLoader(
                val_traj_ds, batch_size=args.batch_size_traj,
                shuffle=False, drop_last=False, num_workers=0,
                collate_fn=trajectory_collate,
            )

    
    contrast_loader = None
    val_contrast_loader = None
    echofocus = None
    if args.contrast:
        assert args.study_embeddings or args.video_embeddings, \
            "--contrast requires --study_embeddings or --video_embeddings"

        if args.video_embeddings:
            from model_echofocus import EchoFocus
            video_embs = load_video_embeddings_by_study(args.video_embeddings)
            input_dim_video = next(iter(video_embs.values())).shape[1]
            echofocus = EchoFocus(
                input_dim=input_dim_video, n_heads=8, ff_dim=input_dim_video, dropout=0.2,
            ).to(device)
            if args.echofocus_checkpoint:
                echofocus.load_state_dict(
                    torch.load(args.echofocus_checkpoint, weights_only=False), strict=False,
                )
                print(f"Loaded EchoFocus checkpoint: {args.echofocus_checkpoint}")
            n_ef_params = sum(p.numel() for p in echofocus.parameters())
            print(f"EchoFocus: {n_ef_params:,} params")

            contrast_ds = VideoContrastDataset(
                manifest_ids, data, video_embs, demo_mean, demo_std,
                n_sample=args.n_sample, n_videos_sample=args.n_videos_sample,
            )
            collate_fn = video_contrast_collate
        else:
            video_embs = load_video_embeddings(args.study_embeddings)
            contrast_ds = ContrastDataset(manifest_ids, data, video_embs, demo_mean, demo_std)
            collate_fn = contrast_collate

        print(f"Contrast pairs: {len(contrast_ds):,} (of {len(manifest_ids):,} studies)")
        contrast_loader = DataLoader(
            contrast_ds, batch_size=args.batch_size_contrast,
            shuffle=True, drop_last=True, num_workers=0,
            collate_fn=collate_fn,
        )
    if args.contrast and args.val_manifest:
        val_ids = [s for s in Path(args.val_manifest).read_text().strip().splitlines() if s in data]
        if args.video_embeddings:
            val_contrast_ds = VideoContrastDataset(
                val_ids, data, video_embs, demo_mean, demo_std,
                n_sample=args.n_sample, n_videos_sample=args.n_videos_sample,
            )
            val_collate_fn = video_contrast_collate
        else:
            val_contrast_ds = ContrastDataset(val_ids, data, video_embs, demo_mean, demo_std)
            val_collate_fn = contrast_collate
        print(f"Val contrast pairs: {len(val_contrast_ds):,}")
        val_contrast_loader = DataLoader(
            val_contrast_ds, batch_size=args.batch_size_contrast,
            shuffle=False, drop_last=False, num_workers=0,
            collate_fn=val_collate_fn,
        )
    # ---- optimizer ----
    params = list(encoder.parameters())
    if echofocus is not None:
        params += list(echofocus.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- train ----
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        if echofocus is not None:
            echofocus.train()
        vic_total, traj_total, con_total, n_vic, n_traj, n_con = 0.0, 0.0, 0.0, 0, 0, 0
        t0 = time.time()

        vicreg_iter = iter(vicreg_loader) if vicreg_loader else None
        traj_iter = iter(traj_loader) if traj_loader else None
        contrast_iter = iter(contrast_loader) if contrast_loader else None
        n_steps = max(
            len(vicreg_loader) if vicreg_loader else 0,
            len(traj_loader) if traj_loader else 0,
            len(contrast_loader) if contrast_loader else 0,
        )

        for step in range(n_steps):
            loss = torch.tensor(0.0, device=device)

            if vicreg_iter is not None:
                try:
                    batch = next(vicreg_iter)
                except StopIteration:
                    vicreg_iter = iter(vicreg_loader)
                    batch = next(vicreg_iter)
                vic_loss, bs, _ = vicreg_step(encoder, batch, device, args.lam, args.mu, args.nu, args.tau, bernoulli=args.bernoulli_p is not None)
                loss = loss + vic_loss
                vic_total += vic_loss.item() * bs
                n_vic += bs

            if traj_iter is not None:
                try:
                    batch = next(traj_iter)
                except StopIteration:
                    traj_iter = iter(traj_loader)
                    batch = next(traj_iter)
                t_loss, bs = trajectory_step(encoder, batch, device)
                loss = loss + args.traj_weight * t_loss
                traj_total += t_loss.item() * bs
                n_traj += bs

            if contrast_iter is not None:
                try:
                    batch = next(contrast_iter)
                except StopIteration:
                    contrast_iter = iter(contrast_loader)
                    batch = next(contrast_iter)
                c_loss, bs = contrast_step(encoder, batch, device, args.contrast_temperature, echofocus)
                loss = loss + args.contrast_weight * c_loss
                con_total += c_loss.item() * bs
                n_con += bs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate
        val_loss = None
        if val_traj_loader is not None:
            val_loss = validate_trajectory(encoder, val_traj_loader, device)

        val_con_loss = None
        if val_contrast_loader is not None:
            val_con_loss = validate_contrast(encoder, val_contrast_loader, device, args.contrast_temperature, echofocus)
        
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        row = dict(epoch=epoch, lr=lr, time=round(elapsed, 1))
        parts = [f"epoch {epoch}/{args.epochs}"]
        if n_vic > 0:
            row["vicreg_loss"] = vic_total / n_vic
            parts.append(f"vic={row['vicreg_loss']:.4f}")
        if n_traj > 0:
            row["traj_loss"] = traj_total / n_traj
            parts.append(f"traj={row['traj_loss']:.4f}")
        if n_con > 0:
            row["contrast_loss"] = con_total / n_con
            parts.append(f"con={row['contrast_loss']:.4f}")
        if val_loss is not None:
            row["val_traj_loss"] = val_loss
            parts.append(f"val_traj={val_loss:.4f}")
        if val_con_loss is not None:
            row["val_contrast_loss"] = val_con_loss
            parts.append(f"val_con={val_con_loss:.4f}")
        
        parts.append(f"lr={lr:.1e}  {elapsed:.0f}s")
        history.append(row)
        print("  ".join(parts), flush=True)

        # Save
        save_dict = {
            "encoder_state_dict": encoder.state_dict(),
            "demo_mean": demo_mean,
            "demo_std": demo_std,
            "config": vars(args),
            "epoch": epoch,
        }
        torch.save(save_dict, Path(args.output_dir) / "latest.pt")
        if echofocus is not None:
            torch.save(echofocus.state_dict(), Path(args.output_dir) / "echofocus_latest.pt")

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(save_dict, Path(args.output_dir) / "best.pt")

    # ---- save results ----
    results = dict(n_params=n_params, history=history, config=vars(args))
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
