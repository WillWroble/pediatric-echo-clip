"""Encode echo reports → embeddings.npz for downstream eval.

Usage:
    python -u embed.py \
        --checkpoint results/v1/latest.pt \
        --h5_dir /lab-share/.../Line_Embeddings \
        --train_manifest manifests/train.txt \
        --val_manifest manifests/val.txt \
        --output results/v1/embeddings.npz
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from report_encoder import ReportEncoder
from report_dataset import preload_all, normalize_demos


@torch.no_grad()
def encode_studies(model, study_ids, data, demo_mean, demo_std, device):
    model.eval()
    embeddings = []
    for sid in study_ids:
        lines, demos, _, _ = data[sid]
        demos_n = normalize_demos(demos, demo_mean, demo_std)
        lines_t = torch.from_numpy(lines).unsqueeze(0).to(device)
        demos_t = torch.from_numpy(demos_n).unsqueeze(0).to(device)
        z = model.encode(lines_t, demos_t).squeeze(0).cpu().numpy()
        embeddings.append(z)
    return np.stack(embeddings).astype(np.float32)

@torch.no_grad()
def encode_videos(model, study_ids, video_embs, device):
    model.eval()
    filtered_ids = [sid for sid in study_ids if sid in video_embs]
    embeddings = []
    for sid in filtered_ids:
        x = torch.from_numpy(video_embs[sid]).unsqueeze(0).to(device)
        h = model.encode(x).squeeze(0).cpu().numpy()
        embeddings.append(h)
    return np.stack(embeddings).astype(np.float32), filtered_ids
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5_dir", required=True)
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--echofocus_checkpoint", default=None, help="EchoFocus checkpoint for video embedding")
    p.add_argument("--video_embeddings", default=None, help=".npz of video-level embeddings")
    p.add_argument("--video_output", default=None, help="Output .npz for video embeddings (same format as text)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    demo_mean, demo_std = ckpt["demo_mean"], ckpt["demo_std"]

    encoder = ReportEncoder(
        input_dim=768, n_heads=cfg["n_heads"], dropout=0.0, n_demo=6,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})", flush=True)

    data = preload_all(args.h5_dir)

    train_ids = [s for s in Path(args.train_manifest).read_text().strip().splitlines() if s in data]
    val_ids   = [s for s in Path(args.val_manifest).read_text().strip().splitlines()   if s in data]
    print(f"Train: {len(train_ids):,}  Val: {len(val_ids):,}", flush=True)

    print("Encoding train...", flush=True)
    Z_train = encode_studies(encoder, train_ids, data, demo_mean, demo_std, device)
    print("Encoding val...", flush=True)
    Z_val = encode_studies(encoder, val_ids, data, demo_mean, demo_std, device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        train=Z_train, val=Z_val,
        train_ids=np.array(train_ids), val_ids=np.array(val_ids),
    )
    print(f"Saved {args.output}  (train: {Z_train.shape}, val: {Z_val.shape})", flush=True)
    if args.echofocus_checkpoint and args.video_embeddings:
        from model_echofocus import EchoFocus
        from report_dataset import load_video_embeddings_by_study

        video_embs = load_video_embeddings_by_study(args.video_embeddings)
        input_dim_video = next(iter(video_embs.values())).shape[1]

        echofocus = EchoFocus(
            input_dim=input_dim_video, n_heads=8, ff_dim=input_dim_video, dropout=0.0,
        ).to(device)
        echofocus.load_state_dict(torch.load(args.echofocus_checkpoint, weights_only=False), strict=False)
        echofocus.eval()
        print(f"Loaded EchoFocus checkpoint", flush=True)

        print("Encoding video (train)...", flush=True)
        V_train, v_train_ids = encode_videos(echofocus, train_ids, video_embs, device)
        print("Encoding video (val)...", flush=True)
        V_val, v_val_ids = encode_videos(echofocus, val_ids, video_embs, device)

        video_out = args.video_output or args.output.replace(".npz", "_video.npz")
        np.savez_compressed(
            video_out,
            train=V_train, val=V_val,
            train_ids=np.array(v_train_ids), val_ids=np.array(v_val_ids),
        )
        print(f"Saved {video_out}  (train: {V_train.shape}, val: {V_val.shape})", flush=True)
if __name__ == "__main__":
    main()
