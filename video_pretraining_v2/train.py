"""DDP training loop for ClipAggregator on frozen EchoJEPA patch tokens.

Uses JEPA's own VideoDataset + IntraStudyBatchSampler grouped on study_id.
Inference-style encode path: resize + ImageNet normalize, no augmentation.
Launch via torchrun from train.sbatch.
"""

import argparse
import math
import os
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.datasets.video_dataset import VideoDataset
from src.datasets.video_dataset import IntraStudyBatchSampler, IntraMRNBatchSampler
from src.hub.backbones import _clean_backbone_key
from src.models import vision_transformer as vit

from model import ClipAggregator


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


# ---------------------------------------------------------------------------
# DDP utilities
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def all_gather_with_grad(x):
    world = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


def all_gather_nograd(x):
    world = dist.get_world_size()
    gathered = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)


# ---------------------------------------------------------------------------
# Loss and schedule
# ---------------------------------------------------------------------------

def infonce_loss(z, sids, temperature=0.1):
    sim = z @ z.T / temperature
    pos_mask = sids.unsqueeze(0) == sids.unsqueeze(1)
    pos_mask.fill_diagonal_(False)

    sim.fill_diagonal_(float("-inf"))
    log_denom = torch.logsumexp(sim, dim=1)

    pos_per_anchor = pos_mask.sum(dim=1).clamp(min=1)
    pos_sim = torch.where(pos_mask, sim, torch.zeros_like(sim)).sum(dim=1)
    loss_per_anchor = log_denom - pos_sim / pos_per_anchor

    has_pos = pos_mask.any(dim=1)
    loss = loss_per_anchor[has_pos].mean() if has_pos.any() else torch.tensor(0.0, device=z.device)
    return loss, int(pos_mask.sum().item())


def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class ClipTransform:
    """Resize + ImageNet normalize. (T, H, W, 3) uint8 -> (3, T, H, W) float."""
    def __init__(self, resolution=224):
        self.resolution = resolution

    def __call__(self, buffer):
        x = torch.from_numpy(buffer).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
        x = F.interpolate(x, size=(self.resolution, self.resolution),
                          mode="bilinear", align_corners=False)
        x = x.permute(1, 0, 2, 3)  # (3, T, H, W)
        return (x - MEAN) / STD


def clip_collate(batch):
    """VideoDataset returns (buffer_list, label, clip_indices). num_clips=1 -> list of 1."""
    clips = torch.stack([b[0][0] for b in batch])  # (B, 3, T, H, W)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return clips, labels


def load_sid_map(sid_map_path):
    """Load int -> str study_id mapping written by ensure_pretrain_csv."""
    mapping = {}
    with open(sid_map_path) as f:
        for line in f:
            i, sid = line.rstrip("\n").split("\t")
            mapping[int(i)] = sid
    return mapping
def ensure_pretrain_csv(avi_manifest, pretrain_manifest, output_csv):
    """Rank-0-only. Generates space-delimited CSV with rows `path study_id_int`."""
    output_csv = Path(output_csv)
    if output_csv.exists():
        return
    keep = set(Path(pretrain_manifest).read_text().split())
    sid_to_int = {}
    lines = []
    with open(avi_manifest) as f:
        for line in f:
            path = line.split(maxsplit=1)[0]
            sid = path.split("/")[-2]
            if sid.endswith("_trim"):
                sid = sid[:-5]
            if sid not in keep:
                continue
            if sid not in sid_to_int:
                sid_to_int[sid] = len(sid_to_int)
            lines.append(f"{path} {sid_to_int[sid]}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("\n".join(lines) + "\n")
    with open(output_csv.with_suffix(".sid_map.txt"), "w") as f:
        for sid, i in sorted(sid_to_int.items(), key=lambda x: x[1]):
            f.write(f"{i}\t{sid}\n")
    print(f"wrote {output_csv}: {len(lines):,} videos, {len(sid_to_int):,} studies", flush=True)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def load_frozen_jepa_encoder(checkpoint_path, device):
    model = vit.vit_base(
        img_size=(224, 224),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        use_rope=True,
        use_sdpa=True,
        uniform_power=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("target_encoder") or ckpt.get("encoder")
    if state is None:
        raise KeyError(f"no encoder in {checkpoint_path}: keys={list(ckpt.keys())}")
    state = _clean_backbone_key(state)
    msg = model.load_state_dict(state, strict=False)
    print(f"  encoder loaded: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}",
          flush=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--avi_manifest", required=True)
    p.add_argument("--pretrain_manifest", required=True)
    p.add_argument("--encoder_checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--m_studies", type=int, default=2048)
    p.add_argument("--k_clips", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--frames_per_clip", type=int, default=16)
    p.add_argument("--frame_step", type=int, default=2)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--intra_mrn", action="store_true")
    p.add_argument("--mrn_csv", default="/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echo_reports_v3.csv")
    args = p.parse_args()

    rank, world, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    is_main = rank == 0

    out = Path(args.output_dir)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        print(f"world={world} m_studies={args.m_studies} k_clips={args.k_clips} "
              f"batch={args.m_studies * args.k_clips}", flush=True)

    # --- CSV generation (rank 0 only) ----------------------------------------
    csv_path = out / "pretrain_videos.csv"
    if is_main:
        ensure_pretrain_csv(args.avi_manifest, args.pretrain_manifest, csv_path)
    dist.barrier()

    # --- Dataset --------------------------------------------------------------
    transform = ClipTransform(resolution=args.resolution)
    dataset = VideoDataset(
        data_paths=[str(csv_path)],
        frames_per_clip=args.frames_per_clip,
        #frame_step=args.frame_step,
        frame_step=None,
        fps=8,
        num_clips=1,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        transform=transform,
    )
    if is_main:
        print(f"dataset: {len(dataset):,} videos", flush=True)

    
    if args.intra_mrn:
        import pandas as pd
        sid_map = load_sid_map(csv_path.with_suffix(".sid_map.txt"))
        str_to_int = {v: k for k, v in sid_map.items()}

        mrn_df = pd.read_csv(args.mrn_csv, usecols=["study_id", "mrn"], dtype=str)
        mrn_df["mrn"] = mrn_df["mrn"].str.lstrip("0")
        sid_to_mrn_str = mrn_df.groupby("study_id")["mrn"].agg(lambda x: x.mode()[0]).to_dict()

        mrn_to_int = {m: i for i, m in enumerate(sorted(set(sid_to_mrn_str.values())))}
        int_sid_to_int_mrn = {
            str_to_int[sid]: mrn_to_int[mrn]
            for sid, mrn in sid_to_mrn_str.items()
            if sid in str_to_int
        }

        max_sid = max(int_sid_to_int_mrn.keys()) + 1
        mrn_lookup = torch.full((max_sid,), -1, dtype=torch.long)
        for sid, mrn in int_sid_to_int_mrn.items():
            mrn_lookup[sid] = mrn
        mrn_lookup = mrn_lookup.to(device)

        sampler = IntraMRNBatchSampler(
            labels=dataset.labels,
            sid_to_mrn=int_sid_to_int_mrn,
            m_mrns=args.m_studies // world,
            num_replicas=world,
            rank=rank,
        )
    else:
        sampler = IntraStudyBatchSampler(
            labels=dataset.labels,
            m_studies=args.m_studies // world,
            k_clips=args.k_clips,
            num_replicas=world,
            rank=rank,
        )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=1,
        collate_fn=clip_collate,
    )

    # --- Models ---------------------------------------------------------------
    if is_main:
        print("loading frozen encoder...", flush=True)
    encoder = load_frozen_jepa_encoder(args.encoder_checkpoint, device=device)

    if is_main:
        print("building aggregator...", flush=True)
    model = ClipAggregator(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * args.epochs
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable params: {n_params/1e6:.2f}M  "
              f"steps/epoch: {steps_per_epoch}  total: {total_steps}", flush=True)

    # --- Resume ---------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        if is_main:
            print(f"resumed from epoch {start_epoch} step {global_step}", flush=True)

    # --- Train ----------------------------------------------------------------
    model.train()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_pos = 0
        n_batches = 0

        for clips, sids in loader:
            clips = clips.to(device, non_blocking=True)
            sids = sids.to(device, non_blocking=True)

            lr = cosine_lr(global_step, total_steps, args.lr, args.warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            #with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            #    patches = encoder(clips)  # (B, 1568, 768)

            
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                B = clips.shape[0]
                patches = torch.empty(B, 1568, 768, dtype=torch.bfloat16, device=device)
                chunk = 32
                for i in range(0, B, chunk):
                    patches[i:i+chunk] = encoder(clips[i:i+chunk])



            
            del clips

            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_local = model(patches)

            z_all = all_gather_with_grad(z_local).float()
            sids_all = all_gather_nograd(sids)

            if args.intra_mrn:
                sids_all = mrn_lookup[sids_all]
            loss, n_pos = infonce_loss(z_all, sids_all, args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pos += n_pos
            n_batches += 1
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                print(f"  [ep {epoch} step {global_step}] loss={loss.item():.4f} "
                      f"pos={n_pos} lr={lr:.2e}", flush=True)

            del sids, patches, z_local, z_all, sids_all, loss

        if is_main:
            avg_loss = epoch_loss / max(1, n_batches)
            avg_pos = epoch_pos / max(1, n_batches)
            print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  avg_pos={avg_pos:.0f}",
                  flush=True)
            torch.save({
                "model": model.module.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "args": vars(args),
            }, out / "latest.pt")

    dist.destroy_process_group()
    if is_main:
        print("done.", flush=True)


if __name__ == "__main__":
    main()
