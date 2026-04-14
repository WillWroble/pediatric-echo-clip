"""Score all pool lines against each eval study via cross-attention.

Same output format as generate.py: hotspots + reference.

Usage:
    python -u generate_lines.py \
        --checkpoint results/v1/latest.pt \
        --video_embeddings /path/to/infonce_768_all.npz \
        --h5_path /path/to/study_findings.h5 \
        --pool_manifest /path/to/train_modern.txt \
        --eval_manifest /path/to/eval_manifest.txt \
        --output results/v1/line_scores.json
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from model import LineEncoder, CrossAttentionPool


def load_video_embeddings_by_study(npz_path):
    data = np.load(npz_path)
    embs, sids = data["embeddings"], data["study_ids"].astype(str)
    by_study = {}
    for emb, sid in zip(embs, sids):
        by_study.setdefault(sid, []).append(emb)
    return {k: np.stack(v).astype(np.float32) for k, v in by_study.items()}


def collect_pool_lines(h5_path, manifest_path, line_filters=None):
    patterns = []
    if line_filters:
        patterns = [re.compile(l.strip(), re.IGNORECASE)
                    for l in open(line_filters) if l.strip() and not l.startswith("#")]
    
    manifest = set(str(int(float(x))) for x in Path(manifest_path).read_text().strip().splitlines())
    
    unique = set()
    with h5py.File(h5_path, "r") as f:
        for sid_raw in f.keys():
            sid = str(int(float(sid_raw)))
            if sid not in manifest:
                continue
            lines = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f[sid_raw][()]]
            if patterns:
                lines = [l for l in lines if not any(p.search(l) for p in patterns)]
            unique.update(lines)
    
    print(f"Collected {len(unique):,} unique pool lines", flush=True)
    return sorted(unique)


def load_study_lines(h5_path, study_ids, line_filters=None):
    patterns = []
    if line_filters:
        patterns = [re.compile(l.strip(), re.IGNORECASE)
                    for l in open(line_filters) if l.strip() and not l.startswith("#")]
    study_lines = {}
    ids_set = set(study_ids)
    with h5py.File(h5_path, "r") as f:
        for sid_raw in f.keys():
            sid = str(int(float(sid_raw)))
            if sid not in ids_set:
                continue
            lines = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f[sid_raw][()]]
            if patterns:
                lines = [l for l in lines if not any(p.search(l) for p in patterns)]
            study_lines[sid] = lines
    return study_lines


@torch.no_grad()
def encode_lines(lines, encoder, tokenizer, device, batch_size=512):
    encoder.eval()
    all_embs = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        embs = encoder(tokens.input_ids.to(device), tokens.attention_mask.to(device))
        all_embs.append(embs.cpu())
        if (i // batch_size + 1) % 100 == 0:
            print(f"  Encoded {i + len(batch):,}/{len(lines):,}", flush=True)
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def score_study(line_embs, videos, pool, device):
    line_embs = line_embs.to(device)
    videos_t = torch.from_numpy(videos).to(device)
    
    Q = pool.W_Q(line_embs)
    K = pool.W_K(videos_t)
    attn = (Q @ K.T) * pool.scale
    attended = attn.softmax(dim=-1) @ videos_t
    
    logits = (line_embs * attended).sum(dim=-1)
    return torch.sigmoid(logits).cpu().numpy()


def find_hotspots(scores, line_embs, threshold=0.3, linkage_cutoff=0.85):
    active = np.where(scores > threshold)[0]
    if len(active) == 0:
        return []
    if len(active) == 1:
        return [[(int(active[0]), float(scores[active[0]]))]]

    # Cosine sim on active lines only
    active_embs = line_embs[active].numpy()
    norms = np.linalg.norm(active_embs, axis=1, keepdims=True) + 1e-8
    active_norm = active_embs / norms
    sub_sim = active_norm @ active_norm.T

    adjacency = (sub_sim >= linkage_cutoff).astype(np.int32)
    np.fill_diagonal(adjacency, 0)

    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)

    hotspots = []
    for c in range(n_components):
        members = active[labels == c]
        ranked = sorted([(int(m), float(scores[m])) for m in members], key=lambda x: -x[1])
        hotspots.append(ranked)

    hotspots.sort(key=lambda h: -h[0][1])
    return hotspots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video_embeddings", required=True)
    p.add_argument("--h5_path", required=True)
    p.add_argument("--pool_manifest", required=True)
    p.add_argument("--eval_manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--line_filters", default=None)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--linkage_cutoff", type=float, default=0.85)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    encoder = LineEncoder().to(device)
    pool = CrossAttentionPool(dim=768).to(device)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    pool.load_state_dict(ckpt["attn_pool"])
    encoder.eval()
    pool.eval()
    print("Loaded checkpoint", flush=True)

    # Collect and encode pool lines once
    pool_lines = collect_pool_lines(args.h5_path, args.pool_manifest, args.line_filters)
    line_embs = encode_lines(pool_lines, encoder, tokenizer, device)
    print(f"Encoded {len(pool_lines):,} pool lines", flush=True)

    # Load videos
    video_embs = load_video_embeddings_by_study(args.video_embeddings)
    print(f"Loaded {len(video_embs):,} studies with video embeddings", flush=True)

    # Eval studies
    eval_ids = set(str(int(float(x))) for x in Path(args.eval_manifest).read_text().strip().splitlines())
    ids = [s for s in video_embs.keys() if s in eval_ids]
    print(f"Eval manifest: {len(ids):,} studies", flush=True)

    # Load reference lines for eval studies
    study_lines = load_study_lines(args.h5_path, ids, args.line_filters)

    # Build line -> index mapping
    line_to_idx = {l: i for i, l in enumerate(pool_lines)}

    results = []
    for i, sid in enumerate(ids):
        scores = score_study(line_embs, video_embs[sid], pool, device)
        hotspots = find_hotspots(scores, line_embs, args.threshold, args.linkage_cutoff)

        hotspot_data = []
        for h_members in hotspots:
            top = h_members[:args.top_k]
            hotspot_data.append([
                {"score": round(score, 4), "text": pool_lines[idx]}
                for idx, score in top
            ])

        # Reference: score actual lines from this study
        ref_lines = study_lines.get(sid, [])
        ref_data = []
        n_unmapped = 0
        for line in ref_lines:
            idx = line_to_idx.get(line)
            if idx is not None:
                ref_data.append({"text": line, "score": round(float(scores[idx]), 4)})
            else:
                n_unmapped += 1

        top_scores = np.sort(scores)[::-1]

        results.append({
            "study_id": sid,
            "n_hotspots": len(hotspot_data),
            "n_active": int((scores > args.threshold).sum()),
            "n_unmapped": n_unmapped,
            "top_5_scores": [round(float(s), 4) for s in top_scores[:5]],
            "hotspots": hotspot_data,
            "reference": ref_data,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ids)}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    total_unmapped = sum(r["n_unmapped"] for r in results)
    total_ref = sum(len(r["reference"]) + r["n_unmapped"] for r in results)
    print(f"Saved {len(results)} studies to {args.output}", flush=True)
    print(f"Unmapped: {total_unmapped}/{total_ref} ({100*total_unmapped/max(total_ref,1):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
