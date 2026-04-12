"""Extract clip embeddings for all studies in a manifest.

Uses JEPA's VideoDataset with num_clips=num_segments and random_clip_sampling=False
to get deterministic temporal segments. Output schema matches v1 infonce_768_all.npz:
{embeddings: (N, 768), study_ids: (N,)}
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets.video_dataset import VideoDataset

from model import ClipAggregator
from train import (
    ClipTransform,
    ensure_pretrain_csv,
    load_frozen_jepa_encoder,
    setup_ddp,
)


def extract_collate(batch):
    """Flatten num_clips per video into the batch dim. Each video contributes num_clips clips."""
    all_clips = []
    all_labels = []
    for buffer_list, label, _ in batch:
        for clip in buffer_list:
            all_clips.append(clip)
            all_labels.append(label)
    return torch.stack(all_clips), torch.tensor(all_labels, dtype=torch.long)


def load_sid_map(sid_map_path):
    """Load int -> str study_id mapping written by ensure_pretrain_csv."""
    mapping = {}
    with open(sid_map_path) as f:
        for line in f:
            i, sid = line.rstrip("\n").split("\t")
            mapping[int(i)] = sid
    return mapping


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--avi_manifest", required=True)
    p.add_argument("--manifest", required=True,
                   help="study ID list to extract (e.g., study_all.txt)")
    p.add_argument("--encoder_checkpoint", required=True)
    p.add_argument("--model_checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--csv_dir", required=True,
                   help="where to write extract_videos.csv + .sid_map.txt")
    p.add_argument("--num_segments", type=int, default=16)
    p.add_argument("--frames_per_clip", type=int, default=16)
    p.add_argument("--frame_step", type=int, default=2)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=4,
                   help="videos per GPU per step (each contributes num_segments clips)")
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    rank, world, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    is_main = rank == 0

    # --- CSV generation (rank 0 only) ----------------------------------------
    csv_path = Path(args.csv_dir) / "extract_videos.csv"
    if is_main:
        ensure_pretrain_csv(args.avi_manifest, args.manifest, csv_path)
    dist.barrier()

    sid_map = load_sid_map(csv_path.with_suffix(".sid_map.txt"))

    # --- Dataset -------------------------------------------------------------
    transform = ClipTransform(resolution=args.resolution)
    dataset = VideoDataset(
        data_paths=[str(csv_path)],
        frames_per_clip=args.frames_per_clip,
        #frame_step=args.frame_step,
        frame_step=None,
        fps=8,
        num_clips=args.num_segments,
        random_clip_sampling=False,
        allow_clip_overlap=False,
        filter_short_videos=False,
        transform=transform,
    )
    if is_main:
        print(f"dataset: {len(dataset):,} videos x {args.num_segments} segments = "
              f"{len(dataset) * args.num_segments:,} clips", flush=True)

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=1,
        collate_fn=extract_collate,
    )

    # --- Models --------------------------------------------------------------
    encoder = load_frozen_jepa_encoder(args.encoder_checkpoint, device=device)

    model = ClipAggregator().to(device)
    ckpt = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- Extract -------------------------------------------------------------
    my_embs = []
    my_labels = []
    for step, (clips, labels) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        #with torch.autocast("cuda", dtype=torch.bfloat16):
        #    patches = encoder(clips)
        #    h = model.encode(patches)
        
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = 64
            patches = torch.cat(
                [encoder(clips[i:i+chunk]) for i in range(0, clips.shape[0], chunk)],
                dim=0,
            )
            h = model.encode(patches)
            #h = patches.mean(dim=1)   # (B, 1568, 768) -> (B, 768)
        my_embs.append(h.float().cpu().numpy())
        my_labels.append(labels.numpy())

        if is_main and (step + 1) % 50 == 0:
            print(f"  rank0 step {step+1}/{len(loader)}", flush=True)

    embs = np.concatenate(my_embs) if my_embs else np.zeros((0, 768), dtype=np.float32)
    labels_int = np.concatenate(my_labels) if my_labels else np.zeros((0,), dtype=np.int64)
    sids = np.array([sid_map[int(i)] for i in labels_int], dtype=object)

    shard = Path(args.output).with_suffix(f".rank{rank}.npz")
    np.savez(shard, embeddings=embs, study_ids=sids)

    dist.barrier()
    if is_main:
        all_embs, all_sids = [], []
        for r in range(world):
            s = Path(args.output).with_suffix(f".rank{r}.npz")
            d = np.load(s, allow_pickle=True)
            all_embs.append(d["embeddings"])
            all_sids.append(d["study_ids"])
            s.unlink()
        embs = np.concatenate(all_embs)
        sids = np.concatenate(all_sids)
        np.savez(args.output, embeddings=embs, study_ids=sids)
        print(f"saved {len(embs):,} embeddings to {args.output}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
