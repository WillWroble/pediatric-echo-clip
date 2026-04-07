"""
Corpus analysis for Echo Project text data.
Outputs per-field stats to inform tokenization and encoding strategy.
"""

import sys
import re
from collections import Counter
from pathlib import Path
from math import log2

import pandas as pd
import numpy as np
from transformers import AutoTokenizer

BASE = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip")
INPUT = BASE / "Summary_Text/echo_reports.csv"
OUT = BASE / "analysis"

TEXT_FIELDS = ["summary", "study_findings", "history"]


def tokenize_words(text):
    return re.findall(r"[a-zA-Z]{2,}", text.lower())

def zipf(counter, field, out_dir):
    rows = counter.most_common()
    df = pd.DataFrame(rows, columns=["word", "count"])
    df.to_csv(out_dir / f"zipf_{field}.csv", index=False)
    return df


def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * log2(c / total) for c in counter.values() if c > 0)


def ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def analyze_field(series, field, out_dir, bert_tok):
    non_null = series.dropna().astype(str)
    non_empty = non_null[non_null.str.strip() != ""]

    # sparsity
    total = len(series)
    filled = len(non_empty)
    sparsity = 1 - filled / total

    # word-level stats
    all_words = []
    doc_lengths = []
    for text in non_empty:
        words = tokenize_words(text)
        all_words.extend(words)
        doc_lengths.append(len(words))

    word_counter = Counter(all_words)
    vocab_size = len(word_counter)

    # zipf
    zipf(word_counter, field, out_dir)

    # entropy
    h = entropy(word_counter)

    # bigrams / trigrams
    bigram_counter = Counter()
    trigram_counter = Counter()
    for text in non_empty:
        words = tokenize_words(text)
        bigram_counter.update(ngrams(words, 2))
        trigram_counter.update(ngrams(words, 3))

    bi_df = pd.DataFrame(bigram_counter.most_common(500), columns=["bigram", "count"])
    bi_df.to_csv(out_dir / f"bigrams_{field}.csv", index=False)

    tri_df = pd.DataFrame(trigram_counter.most_common(500), columns=["trigram", "count"])
    tri_df.to_csv(out_dir / f"trigrams_{field}.csv", index=False)

    # ClinicalBERT token lengths + subword fragmentation
    token_lengths = []
    total_tokens = 0
    total_subwords = 0
    sample = non_empty.sample(min(10000, len(non_empty)), random_state=42)
    for text in sample:
        words = tokenize_words(text)
        encoded = bert_tok(text, add_special_tokens=True)
        tlen = len(encoded["input_ids"])
        token_lengths.append(tlen)
        total_tokens += len(words)
        total_subwords += tlen - 2  # exclude [CLS] and [SEP]

    token_lengths = np.array(token_lengths)
    frag_ratio = total_subwords / total_tokens if total_tokens > 0 else 0

    return {
        "field": field,
        "total_rows": total,
        "non_empty": filled,
        "sparsity": round(sparsity, 4),
        "vocab_size": vocab_size,
        "total_word_tokens": len(all_words),
        "entropy_bits": round(h, 3),
        "word_len_mean": round(np.mean(doc_lengths), 1) if doc_lengths else 0,
        "word_len_median": round(np.median(doc_lengths), 1) if doc_lengths else 0,
        "word_len_p95": round(np.percentile(doc_lengths, 95), 1) if doc_lengths else 0,
        "word_len_max": max(doc_lengths) if doc_lengths else 0,
        "bert_tok_mean": round(np.mean(token_lengths), 1) if len(token_lengths) else 0,
        "bert_tok_median": round(np.median(token_lengths), 1) if len(token_lengths) else 0,
        "bert_tok_p95": round(np.percentile(token_lengths, 95), 1) if len(token_lengths) else 0,
        "bert_tok_max": int(np.max(token_lengths)) if len(token_lengths) else 0,
        "bert_over_512": round((token_lengths > 512).mean(), 4) if len(token_lengths) else 0,
        "subword_frag_ratio": round(frag_ratio, 3),
    }


def vocab_overlap(df, out_dir):
    vocabs = {}
    for field in TEXT_FIELDS:
        series = df[field].dropna().astype(str)
        words = set()
        for text in series:
            words.update(tokenize_words(text))
        vocabs[field] = words

    rows = []
    fields = [f for f in TEXT_FIELDS if f in vocabs]
    for i, a in enumerate(fields):
        for b in fields[i + 1:]:
            intersection = vocabs[a] & vocabs[b]
            union = vocabs[a] | vocabs[b]
            jaccard = len(intersection) / len(union) if union else 0
            rows.append({
                "field_a": a,
                "field_b": b,
                "vocab_a": len(vocabs[a]),
                "vocab_b": len(vocabs[b]),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": round(jaccard, 4),
            })

    overlap_df = pd.DataFrame(rows)
    overlap_df.to_csv(out_dir / "vocab_overlap.csv", index=False)
    return overlap_df


def main():
    print(f"Loading {INPUT}...")
    df = pd.read_csv(INPUT, low_memory=False)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading ClinicalBERT tokenizer...")
    bert_tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    summaries = []
    for field in TEXT_FIELDS:
        if field not in df.columns:
            print(f"  skipping {field} (not in CSV)")
            continue
        print(f"Analyzing {field}...")
        stats = analyze_field(df[field], field, OUT, bert_tok)
        summaries.append(stats)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "field_summary.csv", index=False)
    print("\n=== Field Summary ===")
    print(summary_df.to_string(index=False))

    print("\nComputing vocabulary overlap...")
    overlap = vocab_overlap(df, OUT)
    print(overlap.to_string(index=False))

    print(f"\nAll outputs saved to {OUT}/")


if __name__ == "__main__":
    main()
