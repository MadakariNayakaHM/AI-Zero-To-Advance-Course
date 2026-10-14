"""Figures for Note 08 (Dot Product, Projections & Least Squares).

Run from this folder:  python figures_08.py
Figures are written to ../images/ with the prefix 08_.
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


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


def arrow(ax, start, end, color, label=None, ls="-"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, linestyle=ls))
    if label:
        mid = (np.array(start) + np.array(end)) / 2
        ax.text(*mid, label, color=color, fontsize=11)


# ---------------------------------------------------------------- 1. projection of u onto v
def projection_geometry():
    u, v = np.array([2.0, 3.0]), np.array([4.0, 0.0])
    t = u @ v / (v @ v)
    p = t * v
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    arrow(ax, (0, 0), v, GREY)
    ax.text(4.05, -0.25, "v = (4, 0)", color=GREY)
    arrow(ax, (0, 0), u, BLUE)
    ax.text(1.0, 2.0, "u = (2, 3)", color=BLUE)
    arrow(ax, (0, 0), p, ORANGE)
    ax.text(0.7, -0.45, "tv = (2, 0),  t = u·v / v·v = 8/16", color=ORANGE)
    ax.plot([u[0], p[0]], [u[1], p[1]], ls="--", color=AQUA, lw=2)
    ax.text(2.1, 1.4, "u − tv = (0, 3)\nperpendicular to v", color=AQUA)
    ax.add_patch(plt.Rectangle((p[0] - 0.2, 0), 0.2, 0.2, fill=False, color=AQUA))  # right angle
    ax.set_title("Board, 10 Oct: the 'shadow' of u on v")
    ax.set_xlim(-0.5, 5); ax.set_ylim(-0.8, 3.6); ax.set_aspect("equal"); ax.grid(alpha=0.3)

    ax = axes[1]
    v = np.array([1.0, 2.0])
    P = np.outer(v, v) / (v @ v)
    rng = np.random.default_rng(3)
    pts = rng.uniform(-3, 3, size=(14, 2))
    line = np.linspace(-3.5, 3.5, 2)
    ax.plot(line * v[0] / np.linalg.norm(v) * 1.2, line * v[1] / np.linalg.norm(v) * 1.2, color=GREY, lw=1.5,
            label="line spanned by v = (1, 2)")
    proj = pts @ P.T
    for a, b in zip(pts, proj):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=AQUA, lw=1, ls="--")
    ax.scatter(*pts.T, color=BLUE, s=25, zorder=3, label="x")
    ax.scatter(*proj.T, color=ORANGE, s=25, zorder=3, label="Px  (P = vvᵀ / vᵀv)")
    ax.set_title("P = vvᵀ/vᵀv sends every point to the line,\nalong a perpendicular")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    save("08_projection_geometry.png")


# ---------------------------------------------------------------- 2. least squares line
def least_squares_fit():
    x = np.array([1, 2, 3, 4.0]); y = np.array([2, 3, 5, 6.0])
    c, m = 0.5, 1.4
    yhat = c + m * x
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    xs = np.linspace(0.5, 4.5, 2)
    ax.plot(xs, c + m * xs, color=ORANGE, lw=2, label="ŷ = 0.5 + 1.4x")
    for xi, yi, fi in zip(x, y, yhat):
        ax.plot([xi, xi], [yi, fi], color=AQUA, lw=2)
        ax.text(xi + 0.06, (yi + fi) / 2, f"{yi - fi:+.1f}", color=AQUA, fontsize=9)
    ax.scatter(x, y, color=BLUE, zorder=3, s=40, label="data")
    ax.set_xlabel("hours studied x"); ax.set_ylabel("score y")
    ax.set_title("Least squares: residuals sum to 0,\nSSE = 0.2, R² = 0.98")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    X = np.column_stack([np.ones(4), x])
    cs = np.linspace(-1.5, 2.5, 200); ms = np.linspace(0.8, 2.0, 200)
    C, M = np.meshgrid(cs, ms)
    SSE = ((y[None, None, :] - (C[..., None] + M[..., None] * x[None, None, :])) ** 2).sum(-1)
    cs_ = ax.contour(C, M, SSE, levels=[0.3, 0.6, 1.2, 2.5, 5, 10, 20], colors=GREY, linewidths=1)
    ax.clabel(cs_, fontsize=7)
    ax.scatter([c], [m], color=ORANGE, s=60, zorder=3)
    ax.annotate("β̂ = (XᵀX)⁻¹Xᵀy = (0.5, 1.4)", (c, m), xytext=(c - 1.6, m + 0.35), color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_xlabel("intercept c"); ax.set_ylabel("slope m")
    ax.set_title("SSE(c, m) is a bowl;\nthe normal equations find its bottom")
    ax.grid(alpha=0.3)
    save("08_least_squares_fit.png")


# ---------------------------------------------------------------- 3. projection onto a plane
def projection_plane():
    A = np.array([[1, 0], [1, 1], [1, 2.0]]); b = np.array([6, 0, 0.0])
    xhat = np.linalg.solve(A.T @ A, A.T @ b)
    p = A @ xhat; e = b - p
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    s, t = np.meshgrid(np.linspace(-0.5, 6.5, 10), np.linspace(-3.5, 0.5, 10))
    P = s[..., None] * A[:, 0] + t[..., None] * A[:, 1]
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], alpha=0.18, color=BLUE)
    ax.quiver(0, 0, 0, *b, color=GREY, lw=2, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, *p, color=ORANGE, lw=2, arrow_length_ratio=0.08)
    ax.quiver(*p, *e, color=AQUA, lw=2, arrow_length_ratio=0.15)
    from matplotlib.lines import Line2D
    ax.scatter(*p, color=ORANGE, s=30)
    ax.legend(handles=[Line2D([], [], color=GREY, lw=2, label="b = (6, 0, 0), not in the plane"),
                       Line2D([], [], color=ORANGE, lw=2, label="p = Ax̂ = (5, 2, −1), closest point in C(A)"),
                       Line2D([], [], color=AQUA, lw=2, label="e = b − p = (1, −2, 1), perpendicular to C(A)")],
              loc="upper left", fontsize=8)
    ax.set_title("Projection onto the column space C(A)\n(the plane spanned by (1,1,1) and (0,1,2))")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-35)
    save("08_projection_plane.png")


if __name__ == "__main__":
    projection_geometry()
    least_squares_fit()
    projection_plane()
