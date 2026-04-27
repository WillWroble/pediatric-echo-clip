"""Evaluate trained EchoFocus on external data.

Usage:
    python -u eval_external.py \
        --checkpoint results/efm_jepa_encode_v1_frac1.0/best.pt \
        --embeddings ../video_pretraining_v2/embeddings/jepa_outside.npz \
        --labels /lab-share/Cardio-Mayourian-e2/Public/Echo_Labels/echo_measurements_090425.csv \
        --label_col EF05 \
        --sid_col eid \
        --scale 100
"""

import argparse
import numpy as np
import torch
import pandas as pd
from model import EchoFocus

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--label_col", default="LVEF")
    p.add_argument("--sid_col", default="Event.ID.Number")
    p.add_argument("--scale", type=float, default=1.0, help="Multiply labels by this (e.g., 100 if 0-1 scale)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load embeddings
    data = np.load(args.embeddings, allow_pickle=True)
    all_embs = data["embeddings"]
    all_sids = data["study_ids"].astype(str)
    input_dim = all_embs.shape[1]

    emb_by_study = {}
    for emb, sid in zip(all_embs, all_sids):
        emb_by_study.setdefault(sid, []).append(emb)
    emb_by_study = {k: np.stack(v, dtype=np.float32) for k, v in emb_by_study.items()}
    print(f"Loaded {len(emb_by_study)} studies, {input_dim}d")

    # Load labels
    df = pd.read_csv(args.labels, low_memory=False)
    df["sid"] = df[args.sid_col].astype(str)
    df = df[["sid", args.label_col]].drop_duplicates(subset="sid").dropna()
    
    # Filter valid range (0-1 for EF05)
    df = df[(df[args.label_col] >= 0) & (df[args.label_col] <= 1)]
    
    available = sorted(set(emb_by_study) & set(df["sid"]))
    labels = (df.set_index("sid")[args.label_col] * args.scale).to_dict()
    print(f"{len(available)} studies with valid labels")

    if len(available) == 0:
        print("No overlap!")
        return

    # Load model
    model = EchoFocus(input_dim=input_dim, n_heads=8, ff_dim=input_dim, dropout=0.0, n_targets=1)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.to(device).eval()

    # Evaluate
    preds, trues = [], []
    for sid in available:
        emb = torch.from_numpy(emb_by_study[sid]).unsqueeze(0).to(device)
        out = model(emb).squeeze()
        preds.append(out.cpu().item() * 100)  # model outputs 0-1
        trues.append(labels[sid])

    preds = np.array(preds)
    trues = np.array(trues)
    
    mae = np.abs(preds - trues).mean()
    ss_res = ((preds - trues) ** 2).sum()
    ss_tot = ((trues - trues.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    r = np.sqrt(max(0, r2))

    print(f"\nExternal eval: {len(available)} studies")
    print(f"MAE: {mae:.2f}")
    print(f"R²:  {r2:.4f}")
    print(f"R:   {r:.4f}")

if __name__ == "__main__":
    main()
