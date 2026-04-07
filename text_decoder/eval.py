import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from transformers import AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def load_ground_truth(h5_path, study_ids):
    """Decode ground truth token IDs to text."""
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    references = {}
    with h5py.File(h5_path, "r") as f:
        for sid in study_ids:
            if sid in f:
                tokens = f[sid][:].astype(np.int64).tolist()
                references[sid] = tokenizer.decode(tokens, skip_special_tokens=True)
    return references


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generated", required=True, help="generated_findings.json")
    p.add_argument("--h5_path", required=True, help="study_findings.h5")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- load ----
    with open(args.generated) as f:
        generated = json.load(f)

    study_ids = [r["study_id"] for r in generated]
    gen_texts = {r["study_id"]: r["generated"] for r in generated}
    ref_texts = load_ground_truth(args.h5_path, study_ids)

    # keep only studies with both
    paired = [(sid, gen_texts[sid], ref_texts[sid])
              for sid in study_ids if sid in ref_texts]
    print(f"Evaluating {len(paired):,} studies", flush=True)

    sids, gens, refs = zip(*paired)

    # ---- BLEU ----
    refs_tok = [[r.split()] for r in refs]
    gens_tok = [g.split() for g in gens]
    smooth = SmoothingFunction().method1
    bleu = corpus_bleu(refs_tok, gens_tok, smoothing_function=smooth)
    print(f"BLEU: {bleu:.4f}", flush=True)

    # ---- ROUGE-L ----
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(r, g)["rougeL"].fmeasure for r, g in zip(refs, gens)]
    rouge_l = np.mean(rouge_scores)
    print(f"ROUGE-L: {rouge_l:.4f}", flush=True)

    # ---- BERTScore ----
    P, R, F1 = bert_score(list(gens), list(refs), lang="en", verbose=True)
    bert_f1 = F1.mean().item()
    print(f"BERTScore F1: {bert_f1:.4f}", flush=True)

    # ---- save ----
    metrics = {
        "n_studies": len(paired),
        "bleu": round(bleu, 4),
        "rouge_l": round(rouge_l, 4),
        "bert_score_f1": round(bert_f1, 4),
    }

    out_path = Path(args.output_dir) / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
