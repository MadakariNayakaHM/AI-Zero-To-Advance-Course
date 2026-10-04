"""Generates every figure used in the Applied Math notes.

Run from this folder:  python make_figures.py
Figures are written to ../images/
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


def axes_through_origin(ax, lim):
    ax.axhline(0, c="k", lw=0.8)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")


def arrow(ax, v, color, label=None, origin=(0, 0), **kw):
    ax.annotate("", xy=(origin[0] + v[0], origin[1] + v[1]), xytext=origin,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, **kw))
    if label:
        ax.text(origin[0] + v[0] * 1.05, origin[1] + v[1] * 1.05, label, color=color, fontsize=10)


x = np.linspace(-1, 4, 100)

# 1. Three cases of a 2x2 system
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
ax[0].plot(x, 3 - 2 * x, label="2x + y = 3")
ax[0].plot(x, (3 - x) / 2, label="x + 2y = 3")
ax[0].scatter([1], [1], c="r", zorder=3, s=50)
ax[0].set_title("Unique solution (1, 1)\nlines intersect, det = 3 ≠ 0")
ax[1].plot(x, 3 - 2 * x, lw=5, alpha=0.4, label="2x + y = 3")
ax[1].plot(x, (6 - 4 * x) / 2, "--", label="4x + 2y = 6")
ax[1].set_title("Infinitely many solutions\nsame line (redundant), det = 0")
ax[2].plot(x, 3 - 2 * x, label="2x + y = 3")
ax[2].plot(x, 4 - 2 * x, label="2x + y = 4")
ax[2].set_title("No solution\nparallel lines (inconsistent), det = 0")
for a in ax:
    a.set_xlim(-1, 4); a.set_ylim(-3, 5); a.legend(fontsize=8); a.set_xlabel("x"); a.set_ylabel("y")
save("01_three_cases.png")

# 2. Reflection about y = x
fig, ax = plt.subplots(figsize=(4.5, 4.5))
axes_through_origin(ax, 5)
ax.plot([-5, 5], [-5, 5], "k--", lw=1, label="mirror y = x")
for p, c in [((1, 4), "C0"), ((3, 1), "C1"), ((-2, 3), "C2")]:
    q = (p[1], p[0])
    ax.scatter(*p, c=c, s=40); ax.scatter(*q, c=c, s=40, marker="s")
    ax.plot([p[0], q[0]], [p[1], q[1]], c=c, ls=":")
    ax.text(p[0] + 0.15, p[1] + 0.15, f"{p}", color=c); ax.text(q[0] + 0.15, q[1] - 0.45, f"{q}", color=c)
ax.set_title("[[0,1],[1,0]] swaps (x, y) → (y, x)\n(reflection; applying twice = identity)")
ax.legend(loc="lower right", fontsize=8)
save("02_reflection.png")

# 3. Rotation (orthogonal matrix) preserves lengths and angles
th = np.deg2rad(30)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
sq = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
fig, ax = plt.subplots(figsize=(4.5, 4.5))
axes_through_origin(ax, 1.8)
ax.fill(*sq, alpha=0.3, label="unit square")
ax.fill(*(R @ sq), alpha=0.4, label="R(30°) · square")
ax.set_title("Orthogonal matrix = rotation/reflection\nRᵀR = I, shape & size preserved")
ax.legend(fontsize=8, loc="lower left")
save("03_rotation_orthogonal.png")

# 4. Shear matrices (elementary row operation R_i <- R_i + k R_j)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, M, t in [(ax[0], np.array([[1, 0], [1, 1]]), "[[1,0],[1,1]]: shear in x₂ direction"),
                (ax[1], np.array([[1, 1], [0, 1]]), "[[1,1],[0,1]]: shear in x₁ direction")]:
    a.fill(*sq, alpha=0.3, label="unit square")
    a.fill(*(M @ sq), alpha=0.4, label="after shear")
    a.set_xlim(-0.5, 2.5); a.set_ylim(-0.5, 2.5); a.set_aspect("equal"); a.set_title(t); a.legend(fontsize=8)
save("04_shear.png")

# 5. Row operation R2 <- R2 + k R1 rotates the line about the solution (1, 1)
fig, ax = plt.subplots(figsize=(5.5, 5))
xx = np.linspace(-0.5, 3.5, 100)
ax.plot(xx, 3 - 2 * xx, lw=3, label="R1: 2x₁ + x₂ = 3")
for k in [0, 1, 2, -1]:
    a1, a2, b = 1 + 2 * k, 2 + k, 3 + 3 * k
    ax.plot(xx, (b - a1 * xx) / a2, label=f"k={k}: {a1}x₁ + {a2}x₂ = {b}")
ax.axvline(1, c="C5", label="k=-2: -3x₁ = -3 (x₁ = 1)")
ax.scatter([1], [1], c="k", s=60, zorder=4)
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-1, 4); ax.set_aspect("equal")
ax.set_title("R₂ ← R₂ + kR₁: every new line\nstill passes through the solution (1, 1)")
ax.legend(fontsize=7, loc="upper right")
save("05_row_operation_lines.png")

# 6. Subspaces of R^2 vs a non-subspace
fig, ax = plt.subplots(figsize=(5, 5))
axes_through_origin(ax, 5)
t = np.linspace(-5, 5, 10)
ax.plot(t, t, label="{(x, x)}: line y = x ✓")
ax.plot(t, 2 * t, label="{(x, 2x)}: line y = 2x ✓")
ax.plot(t, -t / 3, label="{(−3x, x)} ✓")
ax.plot(t, 0 * t + 3, "r--", lw=2, label="{(x, 3)}: y = 3 ✗ (no origin)")
ax.scatter([0], [0], c="k", s=60, zorder=4, label="{(0, 0)} ✓ (0-D)")
ax.set_title("Subspaces of ℝ²: {0}, lines through origin, ℝ² itself")
ax.legend(fontsize=7, loc="lower right")
save("06_subspaces_R2.png")

# 7. Span of (2,5) and (1,5): skewed grid covering the whole plane
u, v = np.array([2, 5]), np.array([1, 5])
fig, ax = plt.subplots(figsize=(5.5, 5.5))
axes_through_origin(ax, 7)
for c in range(-6, 7):
    p0, p1 = c * u - 8 * v, c * u + 8 * v
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c="C0", alpha=0.25, lw=0.8)
    p0, p1 = c * v - 8 * u, c * v + 8 * u
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c="C1", alpha=0.25, lw=0.8)
arrow(ax, u, "C0")
arrow(ax, v, "C1")
ax.text(2.3, 4.6, "u=(2,5)", color="C0"); ax.text(-1.9, 5.3, "v=(1,5)", color="C1")
target = np.array([1, 0])
ab = np.linalg.solve(np.c_[u, v], target)
arrow(ax, ab[0] * u, "C0", origin=(0, 0), ls="--")
arrow(ax, ab[1] * v, "C1", origin=tuple(ab[0] * u), ls="--")
ax.scatter(*target, c="r", s=60, zorder=4)
ax.text(target[0] + 0.3, target[1] - 0.8, f"(1,0) = {ab[0]:.0f}·u {ab[1]:+.0f}·v", color="r", fontsize=9)
ax.set_title("Every point of ℝ² = αu + βv\n(u, v independent → basis of ℝ²)")
save("07_span_basis.png")

# 8. Linear dependence: w = 2e1 + 3e2
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5); ax.set_aspect("equal")
arrow(ax, (1, 0), "C0", "e₁=(1,0)")
arrow(ax, (0, 1), "C1", "e₂=(0,1)")
arrow(ax, (2, 3), "C3", "w=(2,3)")
arrow(ax, (2, 0), "C0", origin=(0, 0), ls=":")
arrow(ax, (0, 3), "C1", origin=(2, 0), ls=":")
ax.set_title("2e₁ + 3e₂ − w = 0\n→ {e₁, e₂, w} is linearly DEPENDENT")
save("08_linear_dependence.png")

# 9. Null spaces: line in R^2 and line in R^3
fig = plt.figure(figsize=(10, 4.5))
a1 = fig.add_subplot(1, 2, 1)
axes_through_origin(a1, 3)
a1.plot(t, -t, lw=3, c="C3", label="Null(A) = {k(1, −1)}")
arrow(a1, (1, -1), "k", "(1,−1)")
a1.set_title("A = [[1,1],[1,1]]: null space is a line\n(1-D subspace of ℝ², nullity 1)")
a1.legend(fontsize=8)
a2 = fig.add_subplot(1, 2, 2, projection="3d")
s = np.linspace(-1.5, 1.5, 20)
d = np.array([-3, 1, 1])
a2.plot(*(np.outer(s, d).T), c="C3", lw=3, label="Null(A) = {t(−3, 1, 1)}")
P, Q = np.meshgrid(np.linspace(-4, 4, 10), np.linspace(-4, 4, 10))
for row, c in [((1, 1, 2), "C0"), ((1, 0, 3), "C2")]:
    Z = -(row[0] * P + row[1] * Q) / row[2]
    a2.plot_surface(P, Q, Z, alpha=0.2, color=c)
a2.set_title("A = [[1,1,2],[1,0,3]]: two planes through 0\nmeet in a line = null space (nullity 1)")
a2.legend(fontsize=8)
save("09_null_space.png")
