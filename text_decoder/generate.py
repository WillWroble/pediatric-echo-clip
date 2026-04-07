import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from model import ReportDecoder


def load_embeddings(npz_path, split):
    npz = np.load(npz_path)
    embs = npz[split].astype(np.float32)
    ids = npz[f"{split}_ids"].astype(str).tolist()
    return ids, embs

@torch.no_grad()
def generate(model, cond, tokenizer, max_seq_len=512, temperature=1.0, top_p=1.0):
    """Autoregressive decoding from a single study embedding."""
    device = cond.device
    tokens = [tokenizer.cls_token_id]

    for _ in range(max_seq_len - 1):
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(x, cond)
        next_logits = logits[0, -1] / temperature

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum - probs > top_p
            sorted_logits[mask] = float("-inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token = sorted_idx[torch.multinomial(probs, 1)].item()
        else:
            next_token = next_logits.argmax().item()

        tokens.append(next_token)
        if next_token == tokenizer.sep_token_id:
            break

    return tokenizer.decode(tokens, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--h5_path", default=None)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model ----
    model = ReportDecoder(max_seq_len=args.max_seq_len).to(device)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # ---- data ----
    study_ids, embs = load_embeddings(args.embeddings, args.split)
    print(f"Generating for {len(study_ids):,} studies ({args.split})", flush=True)
    if args.n_samples is not None and args.n_samples < len(study_ids):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(study_ids), args.n_samples, replace=False)
        study_ids = [study_ids[i] for i in idx]
        embs = embs[idx]

    
    
    
    references = {}
    if args.h5_path:
        import h5py
        with h5py.File(args.h5_path, "r") as f:
            for sid in study_ids:
                if sid in f:
                    references[sid] = tokenizer.decode(f[sid][:].astype(np.int64).tolist(), skip_special_tokens=True)
    # ---- generate ----
    results = []
    for i, (sid, emb) in enumerate(zip(study_ids, embs)):
        cond = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        text = generate(model, cond, tokenizer, args.max_seq_len, args.temperature, args.top_p)
        #results.append({"study_id": sid, "generated": text})
        entry = {"study_id": sid, "generated": text}
        if sid in references:
            entry["reference"] = references[sid]
        results.append(entry)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(study_ids)}", flush=True)

    out_path = Path(args.output_dir) / "generated_findings.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path} ({len(results):,} studies)", flush=True)


if __name__ == "__main__":
    main()
