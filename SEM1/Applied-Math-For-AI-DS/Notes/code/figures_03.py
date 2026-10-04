"""Figures for note 03 (Vector Spaces & Subspaces, deep dive).

Run from this folder:  python figures_03.py
Writes ../images/03x_*.png  (prefix 03x_ avoids clashing with the shared figures).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


def plane_vs_affine():
    """Plane x+y+z=0 (a subspace) vs the parallel plane x+y+z=1 (affine, not a subspace)."""
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    s = np.linspace(-1.5, 1.5, 12)
    X, Y = np.meshgrid(s, s)
    ax.plot_surface(X, Y, -X - Y, alpha=0.35, color="tab:blue")
    ax.plot_surface(X, Y, 1 - X - Y, alpha=0.35, color="tab:red")
    ax.scatter([0], [0], [0], color="k", s=40)
    ax.text(0.05, 0.05, 0.15, "origin", color="k")
    # shift vector p = (1,0,0) maps W onto the affine plane
    ax.quiver(0, 0, 0, 1, 0, 0, color="tab:green", lw=2, arrow_length_ratio=0.15)
    ax.text(1.0, 0.05, 0.1, "p = (1,0,0)", color="tab:green")
    # normal vector
    ax.quiver(0, 0, 0, 0.6, 0.6, 0.6, color="k", lw=1.5, arrow_length_ratio=0.2)
    ax.text(0.65, 0.65, 0.7, "normal (1,1,1)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Blue: x+y+z=0 (subspace, contains 0)\nRed: x+y+z=1 = p + W (affine, misses 0)")
    ax.view_init(elev=18, azim=-60)
    save("03x_plane_vs_affine.png")


def union_vs_sum():
    """Union of the two axes is not closed under +; the sum U+W is all of R^2."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axs:
        ax.axhline(0, c="k", lw=0.8)
        ax.axvline(0, c="k", lw=0.8)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
    t = np.linspace(-3, 3, 2)
    # intersection of y=x and y=-x
    ax = axs[0]
    ax.plot(t, t, c="tab:blue", lw=2, label="U: y = x")
    ax.plot(t, -t, c="tab:orange", lw=2, label="W: y = -x")
    ax.scatter([0], [0], c="red", s=60, zorder=5, label="U ∩ W = {0}")
    ax.set_title("Intersection: always a subspace")
    ax.legend(loc="lower right", fontsize=8)
    # union of axes not closed
    ax = axs[1]
    ax.plot(t, 0 * t, c="tab:blue", lw=3, label="U: x-axis")
    ax.plot(0 * t, t, c="tab:orange", lw=3, label="W: y-axis")
    ax.quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=1, color="tab:blue")
    ax.quiver(0, 0, 0, 1, angles="xy", scale_units="xy", scale=1, color="tab:orange")
    ax.quiver(0, 0, 1, 1, angles="xy", scale_units="xy", scale=1, color="red")
    ax.text(1.05, 1.05, "(1,1) not in U ∪ W", color="red")
    ax.set_title("Union: (1,0)+(0,1) escapes")
    ax.legend(loc="lower right", fontsize=8)
    # sum fills the plane
    ax = axs[2]
    g = np.linspace(-2.5, 2.5, 11)
    G1, G2 = np.meshgrid(g, g)
    ax.scatter(G1, G2, s=6, c="tab:green", alpha=0.6, label="u + w fill R²")
    ax.quiver(0, 0, 2, 0, angles="xy", scale_units="xy", scale=1, color="tab:blue")
    ax.quiver(2, 0, 0, 1.5, angles="xy", scale_units="xy", scale=1, color="tab:orange")
    ax.text(2.05, 1.6, "(2,0)+(0,1.5)", fontsize=8)
    ax.set_title("Sum U + W = R² (a subspace)")
    ax.legend(loc="lower right", fontsize=8)
    save("03x_union_vs_sum.png")


def embedding_analogy():
    """Toy 2-D projection (royalty, gender) of hand-made word vectors from the 03 notebook."""
    E = {"king": (0.95, 0.90), "queen": (0.93, -0.88), "man": (0.10, 0.85), "woman": (0.08, -0.90),
         "prince": (0.90, 0.86), "princess": (0.88, -0.87), "boy": (0.05, 0.80), "girl": (0.04, -0.82)}
    fig, ax = plt.subplots(figsize=(6, 5))
    off = {"king": (0.03, 0.04), "prince": (-0.2, 0.04), "man": (0.03, 0.04), "boy": (-0.12, -0.12),
           "queen": (0.04, -0.03), "princess": (-0.27, -0.03), "woman": (0.03, -0.1), "girl": (-0.14, 0.06)}
    for w, (a, b) in E.items():
        ax.scatter(a, b, c="tab:blue")
        ax.text(a + off[w][0], b + off[w][1], w)
    k, m, wo = np.array(E["king"]), np.array(E["man"]), np.array(E["woman"])
    res = k - m + wo
    ax.quiver(*m, *(wo - m), angles="xy", scale_units="xy", scale=1, color="tab:orange", width=0.006)
    ax.quiver(*k, *(wo - m), angles="xy", scale_units="xy", scale=1, color="tab:orange", width=0.006)
    ax.scatter(*res, marker="x", s=120, c="red", zorder=5)
    ax.text(res[0] - 0.42, res[1] - 0.25, "king - man + woman", color="red")
    ax.set_xlabel("feature 1: royalty")
    ax.set_ylabel("feature 2: gender (+male / -female)")
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title("Analogy = parallelogram rule: same offset (woman - man) twice")
    save("03x_embedding_analogy.png")


if __name__ == "__main__":
    plane_vs_affine()
    union_vs_sum()
    embedding_analogy()
