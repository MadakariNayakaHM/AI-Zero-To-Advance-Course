"""Extra figures for Note 05 (Null Space & Nullity).

Run from this folder:  python figures_05.py
Figures are written to ../images/ with the prefix 05x_ (no clash with 05_row_operation_lines.png).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LinearRegression

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


# ---------------------------------------------------------------- 1. four subspaces
def four_subspaces():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, color, title, sub):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=color, ec="white", lw=2, alpha=0.18))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc="none", ec=color, lw=1.5))
        ax.text(x + w / 2, y + h / 2 + 0.2, title, ha="center", va="center", fontsize=11, color="#222")
        ax.text(x + w / 2, y + h / 2 - 0.3, sub, ha="center", va="center", fontsize=9, color=GREY)

    box(0.3, 3.1, 3.2, 2.2, BLUE, "Row space C(Aᵀ)", "dim r")
    box(0.3, 0.5, 3.2, 2.2, ORANGE, "Null space N(A)", "dim n − r")
    box(6.5, 3.1, 3.2, 2.2, BLUE, "Column space C(A)", "dim r")
    box(6.5, 0.5, 3.2, 2.2, ORANGE, "Left null space N(Aᵀ)", "dim m − r")
    ax.text(1.9, 5.75, "Input space ℝⁿ", ha="center", fontsize=12, weight="bold", color="#222")
    ax.text(8.1, 5.75, "Output space ℝᵐ", ha="center", fontsize=12, weight="bold", color="#222")
    ax.text(1.9, 2.9, "orthogonal", ha="center", va="center", fontsize=9, color=GREY)
    ax.text(8.1, 2.9, "orthogonal", ha="center", va="center", fontsize=9, color=GREY)
    ax.annotate("", xy=(6.4, 4.2), xytext=(3.6, 4.2), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.text(5.0, 4.4, "A maps row space onto C(A)\n(one-to-one)", ha="center", fontsize=9, color="#222")
    ax.annotate("", xy=(5.0, 2.95), xytext=(3.6, 1.6), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
    ax.plot(5.0, 2.95, "o", color=ORANGE, ms=8)
    ax.text(5.0, 2.55, "A x = 0", ha="center", fontsize=9, color="#222")
    ax.text(5.0, 0.25, "Example 7 (4×5 matrix): n = 5, m = 4, r = 3 → dims 3, 2, 3, 1",
            ha="center", fontsize=9, color=GREY)
    save("05x_four_subspaces.png")


# ---------------------------------------------------------------- 2. x_p + N(A)
def affine_solution():
    a = np.array([1.0, 2.0]); b = 5.0                      # one equation x1 + 2 x2 = 5
    z = np.array([-2.0, 1.0])                              # spans N(A)
    xp = np.array([5.0, 0.0])                              # particular solution (x2 = 0)
    xmin = a * b / (a @ a)                                 # min-norm solution = (1, 2)
    t = np.linspace(-4, 4, 2)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    line_n = np.outer(np.linspace(-3, 3, 2), z)
    ax.plot(line_n[:, 0], line_n[:, 1], color=ORANGE, lw=2, label="N(A): x₁ + 2x₂ = 0")
    line_s = xp + np.outer(np.linspace(-1.5, 4.5, 2), z)
    ax.plot(line_s[:, 0], line_s[:, 1], color=BLUE, lw=2, label="solutions: x₁ + 2x₂ = 5")
    row = np.outer(np.linspace(-1, 2.6, 2), a)
    ax.plot(row[:, 0], row[:, 1], color=AQUA, lw=1.5, ls="--", label="row space span{(1, 2)}")
    for p, lab, off in [(xp, "x_p = (5, 0)", (0.15, -0.45)), (xmin, "min-norm (1, 2)", (0.2, 0.1)),
                        (xp + 2 * z, "x_p + 2z = (1, 2)", (0.2, -0.45))]:
        ax.plot(*p, "o", color="#222", ms=7, mec="white", mew=2)
    ax.text(5.15, -0.45, "x_p = (5, 0)", fontsize=9)
    ax.text(1.25, 2.1, "x_p + 2z = (1, 2)\n= min-norm solution", fontsize=9)
    ax.annotate("", xy=xp + z, xytext=xp, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5))
    ax.text(4.0, 0.75, "+z", color=ORANGE, fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(-4, 7); ax.set_ylim(-2.5, 4.5)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Complete solution = x_p + N(A): a shifted copy of the null space", fontsize=10)
    ax.grid(alpha=0.25)
    save("05x_affine_solution.png")


# ---------------------------------------------------------------- 3. VIF before / after
def vif_bars():
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(size=n); x2 = rng.normal(size=n); x4 = rng.normal(size=n)
    x3 = x1 + x2 + 0.05 * rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})

    def vif(d):
        return pd.Series({c: 1 / (1 - LinearRegression().fit(d.drop(columns=c), d[c])
                                  .score(d.drop(columns=c), d[c])) for c in d.columns})

    before, after = vif(df), vif(df.drop(columns="x3"))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, s, title in [(axes[0], before, "All four features (x3 ≈ x1 + x2)"),
                         (axes[1], after, "After dropping x3")]:
        ax.bar(s.index, s.values, color=[ORANGE if v > 10 else BLUE for v in s.values], width=0.55)
        for i, v in enumerate(s.values):
            ax.text(i, v * 1.15, f"{v:.1f}", ha="center", fontsize=9, color="#222")
        ax.axhline(10, color=GREY, lw=1, ls="--")
        ax.set_yscale("log"); ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("VIF (log scale)")
    axes[0].text(3.35, 11.5, "rule-of-thumb 10", fontsize=8, color=GREY, ha="right")
    save("05x_vif.png")


if __name__ == "__main__":
    four_subspaces()
    affine_solution()
    vif_bars()
