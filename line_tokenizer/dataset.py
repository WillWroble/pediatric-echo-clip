"""Skip-gram dataset for line-study co-occurrence training."""

import re
import numpy as np
import h5py
import torch
from collections import Counter
from torch.utils.data import Dataset


class SkipGramDataset(Dataset):

    def __init__(self, h5_path, study_ids, study_embs, K=2, M=10, subsample_t=1e-3, line_filters=None):
        """
        Args:
            h5_path:      preprocess output H5 (study_id → line texts).
            study_ids:    list of study IDs (from npz split).
            study_embs:   dict {study_id: 768d numpy array}.
            K:            positive lines per study.
            M:            negative lines per study.
            subsample_t:  word2vec frequency subsampling threshold.
            line_filters: path to regex filter file (lines matching any pattern are dropped).
        """
        self.K = K
        self.M = M

        if line_filters:
            patterns = [re.compile(l.strip(), re.IGNORECASE) for l in open(line_filters)
                        if l.strip() and not l.startswith("#")]
        else:
            patterns = []

        def keep(line):
            return not any(p.search(line) for p in patterns)

        # Load lines per study, normalize IDs
        embs_set = set(study_ids)
        self.study_lines = {}
        with h5py.File(h5_path, "r") as f:
            for sid_raw in f.keys():
                sid = str(int(float(sid_raw)))
                if sid in embs_set:
                    lines = [x.decode("utf-8") if isinstance(x, bytes) else x
                             for x in f[sid_raw][()]]
                    lines = [l for l in lines if keep(l)]
                    if lines:
                        self.study_lines[sid] = lines

        self.study_ids = [s for s in study_ids if s in self.study_lines]
        self.study_embs = study_embs
        print(f"SkipGramDataset: {len(self.study_ids):,} studies", flush=True)

        # Line frequency counts
        counter = Counter()
        for lines in self.study_lines.values():
            counter.update(lines)
        total = sum(counter.values())
        print(f"  {len(counter):,} unique lines, {total:,} total occurrences", flush=True)

        # Subsample keep probability (word2vec formula)
        self.line_keep_prob = {
            line: min(1.0, np.sqrt(subsample_t / (count / total)))
            for line, count in counter.items()
        }

        # Negative sampling pool (freq^0.75 weighting)
        self.all_lines = list(counter.keys())
        freqs = np.array([counter[l] for l in self.all_lines], dtype=np.float64)
        freqs = freqs ** 0.75
        self.neg_probs = freqs / freqs.sum()

        # Pre-tokenize all unique lines
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        all_unique = list(set(l for lines in self.study_lines.values() for l in lines))
        print(f"  Pre-tokenizing {len(all_unique):,} unique lines ...", flush=True)
        enc = tokenizer(all_unique, padding="max_length", truncation=True,
                        max_length=128, return_tensors="np")
        self.token_ids = {}
        self.token_masks = {}
        for i, line in enumerate(all_unique):
            self.token_ids[line] = enc["input_ids"][i]
            self.token_masks[line] = enc["attention_mask"][i]

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        emb = self.study_embs[sid]
        lines = self.study_lines[sid]
        line_set = set(lines)

        # Positive selection with frequency downsampling
        kept = [l for l in lines if np.random.rand() < self.line_keep_prob.get(l, 1.0)]
        if len(kept) < self.K:
            kept = lines
        sel = np.random.choice(len(kept), size=min(self.K, len(kept)), replace=False)
        positives = [kept[i] for i in sel]

        # Negative selection, reject if in anchor study
        negatives = []
        while len(negatives) < self.M:
            i = np.random.choice(len(self.all_lines), p=self.neg_probs)
            candidate = self.all_lines[i]
            if candidate not in line_set:
                negatives.append(candidate)

        all_lines = positives + negatives
        ids = np.stack([self.token_ids[l] for l in all_lines])
        masks = np.stack([self.token_masks[l] for l in all_lines])
        labels = [1.0] * len(positives) + [0.0] * len(negatives)
        return ids, masks, emb, labels


def collate_fn(batch):
    all_ids, all_masks, all_embs, all_labels = [], [], [], []
    for ids, masks, emb, labels in batch:
        all_ids.append(ids)
        all_masks.append(masks)
        all_embs.extend([emb] * len(labels))
        all_labels.extend(labels)
    return (torch.from_numpy(np.concatenate(all_ids)),
            torch.from_numpy(np.concatenate(all_masks)),
            torch.from_numpy(np.stack(all_embs)),
            torch.tensor(all_labels))
