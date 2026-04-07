"""Plot Zipf distribution from summary word frequencies."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ANALYSIS_DIR = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/analysis")
CSV_PATH = ANALYSIS_DIR / "zipf_summary_lines.csv"
OUT_PATH = ANALYSIS_DIR / "zipf_summary_lines_plot.png"

df = pd.read_csv(CSV_PATH)
word_col, freq_col = df.columns[0], df.columns[1]
df = df.sort_values(freq_col, ascending=False).reset_index(drop=True)
df["rank"] = np.arange(1, len(df) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# log-log Zipf plot
ax1.scatter(df["rank"], df[freq_col], s=3, alpha=0.5, color="#457b9d")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Rank (log)")
ax1.set_ylabel("Frequency (log)")
ax1.set_title("Zipf Distribution — Echo Report Summaries")

# annotate top 15 words
for _, row in df.head(15).iterrows():
    ax1.annotate(row[word_col], (row["rank"], row[freq_col]),
                 fontsize=7, alpha=0.8, rotation=25,
                 xytext=(5, 5), textcoords="offset points")

# top 40 bar chart
top = df.head(40)
ax2.barh(np.arange(len(top)), top[freq_col].values, color="#2a9d8f", alpha=0.8)
ax2.set_yticks(np.arange(len(top)))
ax2.set_yticklabels([t[:60] + "..." if len(t) > 60 else t for t in top[word_col].values], fontsize=6)
ax2.invert_yaxis()
ax2.set_xlabel("Frequency")
ax2.set_title("Top 40 Words — Echo Report Summaries")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
print(f"Saved → {OUT_PATH}")
