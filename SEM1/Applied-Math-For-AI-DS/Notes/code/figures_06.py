"""Figures for Note 06 (problem-solving session, column space & rank). Writes ../images/06x_*.png.

Run:  python3 figures_06.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

# 1. Complete solution of x1 + x2 + x3 = 1, x1 + x2 - x3 = -1:  x = (0,0,1) + t(-1,1,0)
fig = plt.figure(figsize=(7, 5.5))
ax = fig.add_subplot(111, projection="3d")
t = np.linspace(-2.5, 2.5, 2)
xh = np.array([-1, 1, 0.0]); xp = np.array([0, 0, 1.0])
null_line = np.outer(t, xh)
sol_line = xp + null_line
ax.plot(*null_line.T, color=BLUE, lw=2.5, label=r"null space $N(A)$: $t(-1,1,0)$")
ax.plot(*sol_line.T, color=ORANGE, lw=2.5, label=r"solutions of $Ax=b$: $x_p + t(-1,1,0)$")
ax.quiver(0, 0, 0, *xp, color=GREEN, lw=2, arrow_length_ratio=0.12)
ax.text(0.05, 0.05, 0.55, r"$x_p=(0,0,1)$", color=GREEN)
pt = xp + (-3) * xh
ax.scatter(*pt, color=ORANGE, s=40)
ax.text(*(pt + [0.1, 0.1, 0.1]), r"$t=-3$: $(3,-3,1)$", color=ORANGE)
ax.plot(*np.outer(np.linspace(-3.5, 3.5, 2), xh).T + 0, alpha=0)  # keep limits symmetric
ax.scatter(0, 0, 0, color=GREY, s=20)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
ax.set_xlim(-3, 3.5); ax.set_ylim(-3.5, 3); ax.set_zlim(-0.5, 1.5)
ax.set_title("Complete solution = particular solution + null space\n(two parallel lines; only the blue one passes through 0)")
ax.legend(loc="upper left", fontsize=9)
ax.view_init(elev=22, azim=-55)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "06x_complete_solution.png"), dpi=150)
plt.close(fig)

# 2. Column space of the rank-2 "food" matrix: plane x - 2y + z = 0
fig = plt.figure(figsize=(7, 5.5))
ax = fig.add_subplot(111, projection="3d")
A = np.array([2, 1, 0.0]); B = np.array([0, 1, 2.0]); C = A + B
s, u = np.meshgrid(np.linspace(-0.5, 2.5, 10), np.linspace(-0.5, 2.0, 10))
P = s[..., None] * A + u[..., None] * B
ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=BLUE, alpha=0.18, edgecolor="none")
for v, name, col in [(A, "food A (2,1,0)", BLUE), (B, "food B (0,1,2)", GREEN), (C, "food C = A + B", GREY)]:
    ax.quiver(0, 0, 0, *v, color=col, lw=2, arrow_length_ratio=0.08)
    ax.text(*(v * 1.05), name, color=col, fontsize=9)
ax.scatter(4, 3, 2, color=GREEN, s=45); ax.text(4.1, 3.1, 2.1, "(4,3,2) reachable", color=GREEN, fontsize=9)
ax.scatter(1, 0, 1, color=ORANGE, s=45); ax.text(1.1, -0.3, 1.25, "(1,0,1) not in C(F)", color=ORANGE, fontsize=9)
ax.set_xlabel("protein"); ax.set_ylabel("carbs"); ax.set_zlabel("fat")
ax.set_title("Column space of a rank-2 3×3 matrix is a plane in ℝ³\n(x − 2y + z = 0): only targets on it can be mixed")
ax.view_init(elev=18, azim=-60)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "06x_column_space_plane.png"), dpi=150)
plt.close(fig)
print("wrote 06x_complete_solution.png, 06x_column_space_plane.png")
