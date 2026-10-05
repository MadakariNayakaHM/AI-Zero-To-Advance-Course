"""Figures for Note 02 (Assignments Guide & Research Toolkit).
Writes images/02_*.png next to the notes.  Run:  python3 Notes/code/figures_02.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})


def h_index(c):
    c = sorted(c, reverse=True)
    return sum(1 for i, x in enumerate(c, 1) if x >= i)


# Figure 1: rank-citation plot with the h-index square
cites = sorted([48, 33, 25, 19, 12, 10, 9, 7, 4, 2, 1, 0], reverse=True)
h = h_index(cites)
r = np.arange(1, len(cites) + 1)
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
ax.bar(r, cites, color=[BLUE if i <= h else "#9ec3ee" for i in r], width=0.8, edgecolor="white", linewidth=2)
ax.plot([0, 13], [0, 13], color=MUTED, lw=1.5, ls="--")
ax.text(12.4, 14.2, "citations = rank", color=MUTED, ha="right")
ax.add_patch(plt.Rectangle((0.5, 0), h, h, fill=False, ec=ORANGE, lw=2))
ax.annotate(f"h = {h}", xy=(h + 0.5, h), xytext=(8.7, 5.2), color=ORANGE, fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE))
ax.set_xticks(r)
ax.set_xlabel("Paper rank (most cited first)")
ax.set_ylabel("Citations")
ax.set_title("h-index: the largest h with h papers of at least h citations", color=INK, loc="left")
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_h_index_rank_plot.png"))
plt.close(fig)

# Figure 2: skewed citation distribution of a synthetic journal (why a mean-based JIF misleads)
rng = np.random.default_rng(2)
c = np.floor(rng.lognormal(mean=1.0, sigma=1.1, size=400)).astype(int)
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
ax.hist(c, bins=np.arange(0, c.max() + 2) - 0.5, color=BLUE, edgecolor="white", linewidth=1)
ax.axvline(c.mean(), color=ORANGE, lw=2)
ax.axvline(np.median(c), color=INK, lw=2, ls="--")
ax.text(c.mean() + 0.5, ax.get_ylim()[1] * 0.9, f"mean = {c.mean():.2f}", color=ORANGE)
ax.text(np.median(c) + 0.5, ax.get_ylim()[1] * 0.75, f"median = {np.median(c):.0f}", color=INK)
ax.set_xlim(-1, 40)
ax.set_xlabel("Citations per article (synthetic journal, 400 articles)")
ax.set_ylabel("Number of articles")
ax.set_title("Citation counts are long-tailed: the mean sits above the typical paper", color=INK, loc="left")
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_citation_distribution.png"))
plt.close(fig)
top10 = np.sort(c)[::-1][:40].sum() / c.sum()
print(f"h={h}; synthetic journal: mean={c.mean():.2f}, median={np.median(c):.0f}, max={c.max()}, "
      f"share of citations from top 10% articles={top10:.3f}, uncited={np.mean(c==0):.3f}")
