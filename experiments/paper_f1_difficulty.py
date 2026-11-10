"""Paper figure: F1 by difficulty for B0, System A and B1 on the test split.

  python -m experiments.paper_f1_difficulty
"""

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.bbox": "tight"})
levels = ["Easy", "Medium", "Hard"]
data = {  # results/phase1/llama-test-*-metrics.json and results/runs/sysA-test-s1/metrics.json
    "B0: no knowledge": [0.655, 0.473, 0.271],
    "System A (source removed)": [0.771, 0.708, 0.608],
    "B1: source given": [0.851, 0.796, 0.615],
}
colors = ["#9aa5b1", "#1f6fb5", "#e08a1e"]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
w = 0.26
for k, (name, vals) in enumerate(data.items()):
    xs = [i + (k - 1) * w for i in range(3)]
    ax.bar(xs, vals, w, label=name, color=colors[k])
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(levels)
ax.set_ylabel("F1 (test split)"); ax.set_ylim(0, 1.0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, columnspacing=0.8)
fig.savefig("results/phase2/figures/f1_by_difficulty_paper.pdf")
