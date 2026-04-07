"""Direct cosine retrieval with contrast projection heads applied post-hoc.

Loads encode() embeddings from embed.py, applies the trained contrast_proj
(GELU -> Linear -> L2-norm) from checkpoints, then evaluates retrieval.
Apples-to-apples comparison with EchoPrime / EchoCLIP retrieval benchmarks.

Usage:
    python -u eval_contrast.py \
        --report_embeddings results/full_contrast_v1/eval/embeddings.npz \
        --video_embeddings  results/full_contrast_v1/eval/embeddings_video.npz \
        --checkpoint        results/full_contrast_v1/latest.pt \
        --echofocus_checkpoint results/full_contrast_v1/echofocus_latest.pt \
        --output_dir        results/full_contrast_v1/eval_contrast
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def apply_contrast_proj(embeddings, state_dict, prefix="contrast_proj"):
    """Apply GELU -> Linear -> L2-norm using saved projection weights."""
    W = state_dict[f"{prefix}.weight"]
    b = state_dict[f"{prefix}.bias"]
    x = torch.from_numpy(embeddings).float()
    x = F.normalize(F.linear(F.gelu(x), W, b), dim=-1)
    return x.numpy()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def recall_at_k(sim, ks=(1, 5, 10, 20, 50, 100)):
    results = {}
    N = sim.shape[0]
    for k in ks:
        if k > N:
            continue
        top_k = np.argpartition(-sim, kth=min(k, N - 1), axis=1)[:, :k]
        hits = sum(i in top_k[i] for i in range(N))
        results[f"R@{k}"] = round(hits / N, 4)
        print(f"  R@{k}: {results[f'R@{k}']:.4f}  ({hits}/{N})")
    return results


def cosine_sim(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(npz, split):
    embs = npz[split].astype(np.float32)
    ids = npz[f"{split}_ids"].astype(str).tolist()
    return embs, ids


def align(r_embs, r_ids, v_embs, v_ids):
    v_map = {sid: i for i, sid in enumerate(v_ids)}
    shared = [sid for sid in r_ids if sid in v_map]
    r_map = {sid: i for i, sid in enumerate(r_ids)}
    R = np.stack([r_embs[r_map[s]] for s in shared])
    V = np.stack([v_embs[v_map[s]] for s in shared])
    return R, V, shared


def plot_bar(results, title, path):
    ks = list(results.keys())
    vals = list(results.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(ks, vals, color="steelblue", alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title(title)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.4f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def eval_retrieval(R, V, ks, label, out):
    """Run both directions and save results."""
    print(f"\n{'='*60}")
    print(f"{label}  (N={len(R):,})")
    print(f"{'='*60}")

    print("\n--- Report -> Video ---")
    r2v = recall_at_k(cosine_sim(R, V), ks)

    print("\n--- Video -> Report ---")
    v2r = recall_at_k(cosine_sim(V, R), ks)

    df = pd.DataFrame({"metric": list(r2v.keys()),
                        "report_to_video": list(r2v.values()),
                        "video_to_report": list(v2r.values())})
    return df, r2v, v2r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report_embeddings", required=True)
    p.add_argument("--video_embeddings", required=True)
    p.add_argument("--checkpoint", required=True, help="Report encoder checkpoint (latest.pt)")
    p.add_argument("--echofocus_checkpoint", default=None, help="EchoFocus checkpoint (echofocus_latest.pt)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--subsample", type=int, default=2000, help="Also eval on a random subsample")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load encode() embeddings
    r_npz = np.load(args.report_embeddings)
    v_npz = np.load(args.video_embeddings)
    r_embs, r_ids = load_split(r_npz, args.split)
    v_embs, v_ids = load_split(v_npz, args.split)

    # Apply contrast projections
    print("Applying report contrast_proj...", flush=True)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    r_embs = apply_contrast_proj(r_embs, ckpt["encoder_state_dict"])
    print(f"  Report: {r_embs.shape}")

    if args.echofocus_checkpoint:
        print("Applying EchoFocus contrast_proj...", flush=True)
        ef_ckpt = torch.load(args.echofocus_checkpoint, weights_only=False, map_location="cpu")
        sd = ef_ckpt if "contrast_proj.weight" in ef_ckpt else ef_ckpt.get("state_dict", ef_ckpt)
        v_embs = apply_contrast_proj(v_embs, sd)
        print(f"  Video: {v_embs.shape}")
    else:
        print("No EchoFocus checkpoint — L2-normalizing video embeddings directly")
        v_embs = v_embs / (np.linalg.norm(v_embs, axis=1, keepdims=True) + 1e-8)

    # Align
    R, V, shared = align(r_embs, r_ids, v_embs, v_ids)
    print(f"\nAligned {len(shared):,} studies", flush=True)

    ks = [k for k in args.ks if k <= len(shared)]

    # Full pool
    df_full, r2v_full, v2r_full = eval_retrieval(R, V, ks, "Full pool", out)
    df_full.to_csv(out / "retrieval_full.csv", index=False)
    plot_bar(r2v_full, f"Report -> Video (N={len(R):,})", out / "r2v_full.png")
    plot_bar(v2r_full, f"Video -> Report (N={len(R):,})", out / "v2r_full.png")

    # Subsample
    if args.subsample and args.subsample < len(shared):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(shared), args.subsample, replace=False)
        R_sub, V_sub = R[idx], V[idx]
        ks_sub = [k for k in args.ks if k <= args.subsample]

        df_sub, r2v_sub, v2r_sub = eval_retrieval(
            R_sub, V_sub, ks_sub, f"{args.subsample:,} subsample", out,
        )
        df_sub.to_csv(out / f"retrieval_{args.subsample}.csv", index=False)
        plot_bar(r2v_sub, f"Report -> Video (N={args.subsample:,})", out / f"r2v_{args.subsample}.png")
        plot_bar(v2r_sub, f"Video -> Report (N={args.subsample:,})", out / f"v2r_{args.subsample}.png")

    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
