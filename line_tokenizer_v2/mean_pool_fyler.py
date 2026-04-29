"""Post-hoc mean-pool AUROC from fyler_scores.npz."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_npz", required=True)
    p.add_argument("--fyler_lines", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data = np.load(args.scores_npz)
    scores = data["scores"]    # (n_studies, n_lines)
    labels = data["labels"]    # (n_studies, n_lines)

    # Rebuild line -> code mapping (same order as columns)
    line_codes, line_texts = [], []
    with open(args.fyler_lines) as f:
        for row in csv.DictReader(f):
            line_codes.append(row["fyler_code"])
            line_texts.append(row["line"])

    # Group line indices by code
    code_to_idxs = defaultdict(list)
    for j, code in enumerate(line_codes):
        code_to_idxs[code].append(j)

    results = []
    for code, idxs in sorted(code_to_idxs.items(), key=lambda x: int(x[0])):
        y = labels[:, idxs[0]]
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            continue

        s_mean = scores[:, idxs].mean(axis=1)
        s_max = scores[:, idxs].max(axis=1)
        s_best_idx = idxs[0]  # will fill in below
        s_best_auc = 0.0

        for j in idxs:
            a = roc_auc_score(y, scores[:, j])
            if a > s_best_auc:
                s_best_auc = a
                s_best_idx = j

        auc_mean = roc_auc_score(y, s_mean)
        auc_max = roc_auc_score(y, s_max)

        results.append({
            "code": code,
            "description": line_texts[idxs[0]],
            "n_pos": n_pos,
            "n_lines": len(idxs),
            "auroc_mean": round(auc_mean, 4),
            "auroc_max": round(auc_max, 4),
            "auroc_best_line": round(s_best_auc, 4),
            "best_line": line_texts[s_best_idx],
        })

    results.sort(key=lambda x: -x["auroc_mean"])

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    aurocs_mean = [r["auroc_mean"] for r in results]
    aurocs_max = [r["auroc_max"] for r in results]
    aurocs_best = [r["auroc_best_line"] for r in results]

    print(f"{len(results)} codes evaluated")
    print(f"  mean-pool:  Mean={np.mean(aurocs_mean):.4f}  Median={np.median(aurocs_mean):.4f}")
    print(f"  max-pool:   Mean={np.mean(aurocs_max):.4f}  Median={np.median(aurocs_max):.4f}")
    print(f"  best-line:  Mean={np.mean(aurocs_best):.4f}  Median={np.median(aurocs_best):.4f}")


if __name__ == "__main__":
    main()
