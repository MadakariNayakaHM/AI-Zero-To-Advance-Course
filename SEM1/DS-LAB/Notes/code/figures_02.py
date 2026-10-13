"""Figures for Note 02 (Pandas DataFrames & data cleaning). Writes ../images/02_*.png.

Run:  python3 figures_02.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#ecebe8"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})

# The running example of the note (same as Part C of 02_pandas_deep_dive.ipynb)
raw = pd.DataFrame({
    "Order_ID":   [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 203, 207, 213, 214],
    "Customer":   ["Anil", "bhavya ", " Chetan", "Deepa", "Esha", "Farhan", "Gita", "Hari",
                   "Isha", "Jay", " Chetan", "Kavya", "Lokesh", "Mani"],
    "City":       ["Dharwad", "Hubli", "dharwad", "Belagavi", "Hubli", "Dharwad ", "Belagavi",
                   "hubli", "Dharwad", "Belagavi", "dharwad", "Hubli", "Dharwad", "HUBLI"],
    "Category":   ["Books", "Electronics", "Books", "Grocery", "Electronics", "Grocery", "Books",
                   "Electronics", "Grocery", "Books", "Books", "Grocery", "Electronics", "Books"],
    "Quantity":   [2, 1, 3, np.nan, 1, 5, 2, 1, 4, np.nan, 3, 6, 1, 2],
    "Unit_Price": [350, 15999, 250, 120, 899, 60, np.nan, 45999, 80, 499, 250, 75, 2499, np.nan],
    "Rating":     [5, 4, np.nan, 3, 7, 4, 5, np.nan, 2, 4, np.nan, 3, 5, 4],
})

# 1. Data-quality map: every cell coloured by its problem
code = np.zeros(raw.shape)                                   # 0 = fine
for j, col in enumerate(raw.columns):
    s = raw[col]
    code[s.isna().to_numpy(), j] = 1                         # missing
    if s.dtype.kind in "OU" or str(s.dtype) == "str":
        messy = (s != s.str.strip().str.title()).to_numpy()
        code[messy, j] = 3                                   # messy text
code[(raw["Rating"] > 5).to_numpy(), raw.columns.get_loc("Rating")] = 2   # invalid
cmap = matplotlib.colors.ListedColormap([GRID, BLUE, ORANGE, AQUA])
fig, ax = plt.subplots(figsize=(8, 5.6))
ax.imshow(code, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
ax.set_xticks(range(raw.shape[1]), raw.columns, rotation=20)
ax.set_yticks(range(raw.shape[0]), [f"row {i}" for i in range(raw.shape[0])])
ax.set_xticks(np.arange(-.5, raw.shape[1]), minor=True)
ax.set_yticks(np.arange(-.5, raw.shape[0]), minor=True)
ax.grid(which="minor", color="white", lw=2)
ax.tick_params(which="minor", length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
for r in (2, 10):
    ax.add_patch(plt.Rectangle((-0.5, r - 0.5), raw.shape[1], 1, fill=False, ec=INK, lw=2, ls="--", zorder=5, clip_on=False))
ax.text(raw.shape[1] - 0.4, 10, "  exact duplicate\n  of row 2", va="center", color=INK, fontsize=9)
for r in (6, 11):
    ax.text(-0.35, r, "ID 207", va="center", ha="left", color=INK, fontsize=8)
ax.legend(handles=[Patch(color=BLUE, label="missing (NaN)"), Patch(color=ORANGE, label="invalid (rating 7)"),
                   Patch(color=AQUA, label="messy text (case / spaces)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
ax.set_title("Where the problems are: 14 orders × 7 columns", color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_data_quality_map.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 2. Mean vs median on a skewed column (after dropping the duplicate row)
price = raw.drop_duplicates()["Unit_Price"].dropna()
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.scatter(price, np.zeros(len(price)), s=60, color=MUTED, zorder=3, ec="white", lw=1.5)
for v, lab, col, y in [(price.median(), f"median = ₹{price.median():,.0f}", BLUE, 0.7),
                       (price.mean(), f"mean = ₹{price.mean():,.0f}", ORANGE, 0.3)]:
    ax.axvline(v, color=col, lw=2)
    ax.text(v, y, " " + lab, color=INK, fontsize=10, va="bottom")
ax.set_ylim(-0.6, 1)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Unit_Price (₹), 11 known values")
ax.set_title("Two expensive items drag the mean far from a typical price", color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_mean_vs_median.png"), dpi=150)
plt.close(fig)

# 3. Mean vs median vs mode on three shapes of data
rng = np.random.default_rng(0)
shapes = {"symmetric": rng.normal(50, 10, 5000),
          "right-skewed (incomes, prices)": rng.lognormal(3.5, 0.6, 5000)}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, (name, x) in zip(axes, shapes.items()):
    ax.hist(x, bins=60, color="#c3c2b7", ec="white")
    ax.axvline(np.mean(x), color=ORANGE, lw=2, label=f"mean {np.mean(x):.1f}")
    ax.axvline(np.median(x), color=BLUE, lw=2, ls="--", label=f"median {np.median(x):.1f}")
    ax.set_title(name, color=INK, loc="left", fontsize=11)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
axes[1].set_xlim(0, 150)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_skew_mean_median.png"), dpi=150)
plt.close(fig)
print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("02_")))
