"""Generate findings via autoregressive centroid prediction with distance-based stopping."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import h5py
import torch

from model import LineDecoder


def load_codebook_labels(codebook_path):
    """Load centroid_id → representative text mapping."""
    cb = np.load(codebook_path, allow_pickle=True)
    return cb["labels"].astype(str).tolist()


def load_reference_texts(h5_path, study_ids):
    """Load ground truth line strings for reference studies."""
    refs = {}
    sid_set = set(study_ids)
    with h5py.File(h5_path, "r") as f:
        for sid in f:
            if sid in sid_set:
                refs[sid] = [x.decode("utf-8") if isinstance(x, bytes) else x
                             for x in f[sid][()]]
    return refs


@torch.no_grad()
def generate(model, study_emb, discrete, max_lines=40, dist_threshold=2.0):
    """Autoregressive generation with distance-based stopping.

    Args:
        model: LineDecoder.
        study_emb: (1, 1, 768) study embedding.
        max_lines: maximum lines to generate.
        dist_threshold: stop if snap distance exceeds this.
    Returns:
        list of centroid indices.
    """
    device = study_emb.device
    predicted_ids = []
    seq = [study_emb.squeeze(0)]  # [(1, 768)]

    for _ in range(max_lines):
        x = torch.stack(seq, dim=1)  # (1, len, 768)
        pred = model(x)  # (1, len, 768)
        next_pred = pred[0, -1:]  # (1, 768)


        if discrete:
            logits = model.head(pred)
            idx = logits[0, -1].argmax().item()
        else:
            idx, dist = model.snap_to_centroid(next_pred)
            if dist.item() > dist_threshold:
                break
            idx = idx.item()
        
        predicted_ids.append(idx)
        seq.append(model.centroids[idx].unsqueeze(0))

    return predicted_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--discrete", action="store_true")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--codebook", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--h5_path", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--max_lines", type=int, default=40)
    p.add_argument("--dist_threshold", type=float, default=2.0)
    p.add_argument("--manifest", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = LineDecoder(codebook_path=args.codebook, max_lines=args.max_lines).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    labels = load_codebook_labels(args.codebook)

    npz = np.load(args.embeddings)
    study_ids = npz[f"{args.split}_ids"].astype(str).tolist()
    embs = npz[args.split].astype(np.float32)

    if args.manifest:
        with open(args.manifest) as f:
            allowed = set(l.strip() for l in f)
        idx = [i for i, s in enumerate(study_ids) if s in allowed]
        study_ids = [study_ids[i] for i in idx]
        embs = embs[idx]

    if args.n_samples and args.n_samples < len(study_ids):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(study_ids), args.n_samples, replace=False)
        study_ids = [study_ids[i] for i in idx]
        embs = embs[idx]

    references = {}
    if args.h5_path:
        print("Loading reference texts...", flush=True)
        references = load_reference_texts(args.h5_path, study_ids)

    print(f"Generating for {len(study_ids):,} studies...", flush=True)
    results = []
    t0 = time.time()

    for i, (sid, emb) in enumerate(zip(study_ids, embs)):
        study_emb = torch.from_numpy(emb).to(device).unsqueeze(0).unsqueeze(0)
        cluster_ids = generate(model, study_emb, args.discrete, args.max_lines, args.dist_threshold)
        gen_lines = [labels[cid] for cid in cluster_ids]

        entry = {
            "study_id": sid,
            "generated_lines": gen_lines,
            "generated_text": "\n".join(gen_lines),
        }
        if sid in references:
            entry["reference_lines"] = references[sid]
            entry["reference_text"] = "\n".join(references[sid])
        results.append(entry)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(study_ids)}  ({elapsed:.0f}s)", flush=True)

    out_path = out / "generated_findings.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path} ({len(results):,} studies)", flush=True)


if __name__ == "__main__":
    main()
