import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/checkpoints")

with open(BASE / "study_embeddings/test_results.json") as f:
    panecho = json.load(f)
with open(BASE / "echofocus_embeddings/test_results.json") as f:
    echofocus = json.load(f)

metrics = ["t2v_R@1", "t2v_R@5", "t2v_R@10", "v2t_R@1", "v2t_R@5", "v2t_R@10"]
labels = ["T→V R@1", "T→V R@5", "T→V R@10", "V→T R@1", "V→T R@5", "V→T R@10"]

pan_vals = [panecho[m] * 100 for m in metrics]
ef_vals = [echofocus[m] * 100 for m in metrics]

x = np.arange(len(metrics))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - w/2, pan_vals, w, label="PanEcho (mean-pooled)", color="#4C72B0")
bars2 = ax.bar(x + w/2, ef_vals, w, label="EchoFocus (LVEF-tuned)", color="#DD8452")

ax.set_ylabel("Recall (%)")
ax.set_title("Contrastive Retrieval: PanEcho vs EchoFocus Video Embeddings\n(ClinicalBERT, 2 layers unfrozen, 20K studies)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, max(pan_vals + ef_vals) * 1.2)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/analysis/embedding_comparison.png", dpi=150)
print("Saved to analysis/embedding_comparison.png")
