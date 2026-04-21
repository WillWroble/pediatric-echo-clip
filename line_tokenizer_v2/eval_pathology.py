"""Evaluate AUROC for canonical pathology lines against val studies."""

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
import torch
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score

from model import LineEncoder, CrossAttentionPool


def merge_soft_wraps(lines):
    if not lines:
        return []
    merged = [lines[0]]
    for line in lines[1:]:
        prev = merged[-1]
        if line[0].islower() or prev.endswith('-'):
            sep = '' if prev.endswith('-') else ' '
            merged[-1] = prev.rstrip('-') + sep + line
        else:
            merged.append(line)
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--h5_path", required=True)
    p.add_argument("--video_embeddings", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_videos", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load anchors
    anchors = []
    for line in Path(args.anchors).read_text().strip().splitlines():
        line = line.strip()#.lstrip("•").strip()
        if line:
            anchors.append(line)
    print(f"Loaded {len(anchors)} anchor lines", flush=True)

    # Load manifest
    manifest = set()
    for x in Path(args.manifest).read_text().strip().splitlines():
        manifest.add(str(int(float(x))))

    # Load video embeddings
    data = np.load(args.video_embeddings)
    embs, sids = data["embeddings"], data["study_ids"].astype(str)
    videos_by_study = {}
    for emb, sid in zip(embs, sids):
        sid = str(int(float(sid)))
        videos_by_study.setdefault(sid, []).append(emb)
    videos_by_study = {k: np.stack(v).astype(np.float32) for k, v in videos_by_study.items()}

    # Load reference lines per study
    ref_lines = {}
    with h5py.File(args.h5_path, "r") as f:
        for sid_raw in f.keys():
            sid = str(int(float(sid_raw)))
            if sid in manifest and sid in videos_by_study:
                lines = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f[sid_raw][()]]
                lines = merge_soft_wraps(lines)
                ref_lines[sid] = set(lines)

    study_ids = list(ref_lines.keys())
    print(f"Loaded {len(study_ids)} studies with videos + reference lines", flush=True)

    # Tokenize anchors
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    enc = tokenizer(anchors, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    anchor_ids = enc["input_ids"].to(device)
    anchor_masks = enc["attention_mask"].to(device)

    # Load model
    encoder = LineEncoder().to(device)
    attn_pool = CrossAttentionPool().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    attn_pool.load_state_dict(ckpt["attn_pool"])
    encoder.eval()
    attn_pool.eval()

    # Encode anchors once
    with torch.no_grad():
        anchor_embs = encoder(anchor_ids, anchor_masks)

    # Build ground truth matrix
    labels = np.zeros((len(study_ids), len(anchors)), dtype=np.float32)
    for i, sid in enumerate(study_ids):
        for j, anchor in enumerate(anchors):
            if anchor in ref_lines[sid]:
                labels[i, j] = 1.0

    print(f"Positives per anchor: {labels.sum(axis=0).astype(int).tolist()}", flush=True)

    # Score all studies
    scores = np.zeros((len(study_ids), len(anchors)), dtype=np.float32)

    with torch.no_grad():
        for i, sid in enumerate(study_ids):
            vids = videos_by_study[sid]
            if vids.shape[0] > args.max_videos:
                idx = np.random.choice(vids.shape[0], args.max_videos, replace=False)
                vids = vids[idx]
            vids_t = torch.from_numpy(vids).unsqueeze(0).to(device)
            mask_t = torch.ones(1, vids_t.shape[1], device=device)

            lines_t = anchor_embs.unsqueeze(0)
            attended = attn_pool(lines_t, vids_t, mask_t)
            logits = (lines_t * attended).sum(dim=-1).squeeze(0)
            scores[i] = logits.cpu().numpy()

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(study_ids)}", flush=True)

    # Compute AUROCs
    results = []
    print("\n" + "=" * 60)
    print(f"{'Anchor':<50} {'N+':>5} {'AUROC':>8}")
    print("=" * 60)
    for j, anchor in enumerate(anchors):
        n_pos = int(labels[:, j].sum())
        if 0 < n_pos < len(study_ids):
            auroc = roc_auc_score(labels[:, j], scores[:, j])
        else:
            auroc = float("nan")
        results.append({"anchor": anchor, "n_pos": n_pos, "n_total": len(study_ids), "auroc": auroc})
        print(f"{anchor:<50} {n_pos:>5} {auroc:>8.3f}")

    valid_aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    print("=" * 60)
    print(f"Mean AUROC: {np.mean(valid_aurocs):.3f} (n={len(valid_aurocs)} anchors)")

    # Save
    with open(Path(args.output_dir) / "aurocs.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(Path(args.output_dir) / "scores.npz",
             scores=scores, labels=labels, anchors=np.array(anchors), study_ids=np.array(study_ids))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
