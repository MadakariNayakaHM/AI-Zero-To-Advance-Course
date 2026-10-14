"""Figures for Note 07 (Linear Transformations).

Run from this folder:  python figures_07.py
Figures are written to ../images/ with the prefix 07_.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

# an "F" shape: asymmetric, so flips and rotations are visible
F_SHAPE = np.array([[0, 0], [0, 2], [1.2, 2], [1.2, 1.6], [0.4, 1.6], [0.4, 1.2],
                    [1, 1.2], [1, 0.8], [0.4, 0.8], [0.4, 0], [0, 0]]).T


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


def draw_grid(ax, M, lim=3):
    """Images of the integer grid lines under M (grey) and of e1, e2 (coloured)."""
    t = np.linspace(-lim, lim, 2)
    for k in range(-lim, lim + 1):
        for line in (np.vstack([t, np.full_like(t, k)]), np.vstack([np.full_like(t, k), t])):
            img = M @ line
            ax.plot(img[0], img[1], color=GREY, lw=0.5, alpha=0.4)
    for vec, col, lab in ((np.array([1, 0]), BLUE, "T(e₁)"), (np.array([0, 1]), ORANGE, "T(e₂)")):
        w = M @ vec
        if np.allclose(w, 0):
            ax.plot(0, 0, "o", color=col, ms=6)
            ax.annotate(lab + " = 0", (0, 0), xytext=(6, -14), textcoords="offset points", color=col, fontsize=9)
        else:
            ax.annotate("", w, (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.text(*(w * 1.1), lab, color=col, fontsize=9)
    ax.fill(*(M @ F_SHAPE), color=AQUA, alpha=0.35, lw=1.2, ec=AQUA)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xticks([]); ax.set_yticks([])


def gallery():
    th = np.deg2rad(30)
    maps = [
        ("identity", np.eye(2)),
        ("scale  diag(2, 0.5)", np.diag([2, 0.5])),
        ("rotate 30°", np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])),
        ("reflect in y = x", np.array([[0, 1], [1, 0]])),
        ("shear  [[1,1],[0,1]]", np.array([[1, 1], [0, 1]])),
        ("project onto x₁-axis  [[1,0],[0,0]]", np.array([[1, 0], [0, 0]])),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    for ax, (name, M) in zip(axes.ravel(), maps):
        draw_grid(ax, M)
        ax.set_title(f"{name}\ndet = {np.linalg.det(M):.2f}", fontsize=10)
    fig.suptitle("Grid lines stay straight, parallel and evenly spaced; the origin stays fixed", fontsize=11)
    save("07_gallery.png")


def class_projection():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    # left: the board picture, (2, y) -> (2, 0)
    for y, col in zip([1, 2, 3], [AQUA, ORANGE, BLUE]):
        a1.annotate("", (2, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        a1.text(2.08, y, f"(2, {y})", color=col, va="center")
    a1.plot([2, 2], [0, 3], ls="--", color=GREY)
    a1.annotate("", (2, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5))
    a1.text(2.05, -0.3, "(2, 0)", fontsize=10)
    a1.set_title("Board, 7 Oct: T(x₁, x₂) = (x₁, 0)\nall three vectors land on (2, 0)")
    a1.set_xlim(-0.3, 5.3); a1.set_ylim(-0.6, 3.5); a1.set_aspect("equal")
    a1.axhline(0, color="k", lw=0.6); a1.axvline(0, color="k", lw=0.6)
    a1.set_xlabel("x₁"); a1.set_ylabel("x₂")
    # right: a line through the origin maps to a line, the x2-axis maps to a point
    t = np.linspace(-1.5, 1.5, 2)
    a2.plot(t, 2 * t, color=BLUE, lw=2, label="line x₂ = 2x₁")
    a2.plot([-3, 3], [0, 0], color=BLUE, lw=5, alpha=0.35, label="its image: the x₁-axis (a line)")
    a2.plot([0, 0], [-3, 3], color=ORANGE, lw=2, label="x₂-axis")
    a2.plot(0, 0, "o", color=ORANGE, ms=10, alpha=0.6, label="its image: the single point 0")
    a2.set_xlim(-3, 3); a2.set_ylim(-3, 3); a2.set_aspect("equal")
    a2.axhline(0, color="k", lw=0.6); a2.axvline(0, color="k", lw=0.6)
    a2.legend(fontsize=8, loc="lower right")
    a2.set_title("Lines through 0 go to lines through 0,\nor collapse to 0 if they lie in the kernel")
    save("07_projection_lines.png")


def affine():
    x = np.linspace(-1, 4, 50)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(x, 2 * x, color=BLUE, lw=2, label="y = 2x  (linear)")
    a1.plot(x, 2 * x + 3, color=ORANGE, lw=2, label="y = 2x + 3  (affine)")
    a1.plot(0, 0, "o", color=BLUE); a1.plot(0, 3, "o", color=ORANGE)
    a1.annotate("f(0) = 3 ≠ 0", (0, 3), xytext=(0.6, 4.6), arrowprops=dict(arrowstyle="->", color=GREY))
    a1.axhline(0, color="k", lw=0.6); a1.axvline(0, color="k", lw=0.6)
    a1.legend(); a1.set_title("Both graphs are straight lines,\nonly one passes through the origin")
    bars = {"f(1) + f(2)": (6, 12), "f(3)": (6, 9)}
    xs = np.arange(2)
    a2.bar(xs - 0.18, [bars[k][0] for k in bars], 0.36, color=BLUE, label="y = 2x")
    a2.bar(xs + 0.18, [bars[k][1] for k in bars], 0.36, color=ORANGE, label="y = 2x + 3")
    for i, k in enumerate(bars):
        a2.text(i - 0.18, bars[k][0] + 0.2, bars[k][0], ha="center")
        a2.text(i + 0.18, bars[k][1] + 0.2, bars[k][1], ha="center")
    a2.set_xticks(xs, list(bars)); a2.legend()
    a2.set_title("Superposition test f(1) + f(2) = f(3)?\n2x passes (6 = 6), 2x + 3 fails (12 ≠ 9)")
    save("07_affine_not_linear.png")


def composition():
    R = np.array([[0, -1], [1, 0]]); F = np.array([[1, 0], [0, -1]])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))
    for ax, (name, M, col) in zip(axes, [("original", np.eye(2), GREY),
                                         ("rotate 90°, then reflect in x-axis\nF·R = [[0,−1],[−1,0]]", F @ R, BLUE),
                                         ("reflect in x-axis, then rotate 90°\nR·F = [[0,1],[1,0]]", R @ F, ORANGE)]):
        ax.fill(*F_SHAPE, color=GREY, alpha=0.15)
        ax.fill(*(M @ F_SHAPE), color=col, alpha=0.5, ec=col)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        if M is not np.eye(2) and name != "original":
            mirror = [2.5, -2.5] if np.array_equal(M, F @ R) else [-2.5, 2.5]
            ax.plot([-2.5, 2.5], mirror, ls=":", color=GREY)
        ax.set_title(name, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Order matters: F·R is a mirror in y = −x, R·F is a mirror in y = x", fontsize=11, y=1.06)
    save("07_composition.png")


if __name__ == "__main__":
    gallery()
    class_projection()
    affine()
    composition()
