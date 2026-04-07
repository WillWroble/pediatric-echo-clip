"""Zipf analysis on full lines across echo report summaries."""

import pandas as pd
from collections import Counter
from pathlib import Path

CSV_PATH = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Summary_Text/echo_reports.csv")
OUT_PATH = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/analysis/zipf_summary_lines.csv")

df = pd.read_csv(CSV_PATH, usecols=["summary"])
df = df.dropna(subset=["summary"])

line_counts = Counter()
for summary in df["summary"]:
    for line in str(summary).split("\n"):
        line = line.strip()
        if line:
            line_counts[line] += 1

out = pd.DataFrame(line_counts.most_common(), columns=["line", "frequency"])
out.to_csv(OUT_PATH, index=False)
print(f"Unique lines: {len(out)}")
print(f"Top 10:")
print(out.head(10).to_string(index=False))
print(f"\nSaved → {OUT_PATH}")
