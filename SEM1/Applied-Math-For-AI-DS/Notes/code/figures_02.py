"""Figures for Note 02 (Gaussian elimination deep dive).

Run from this folder:  python figures_02.py
Writes ../images/02x_*.png  (prefix 02x_ avoids clashing with the shared figures).
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
BLUE, ORANGE = "#2a78d6", "#eb6834"


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


def naive_2x2(eps):
    """Eliminate with the tiny pivot eps (no row swap)."""
    m = 1.0 / eps
    x2 = (2.0 - m * 1.0) / (1.0 - m * 1.0)
    x1 = (1.0 - x2) / eps
    return np.array([x1, x2])


def pivot_2x2(eps):
    """Swap first (partial pivoting): pivot is 1, multiplier is eps."""
    m = eps
    x2 = (1.0 - m * 2.0) / (1.0 - m * 1.0)
    x1 = 2.0 - x2
    return np.array([x1, x2])


# Figure 1: error of x1 vs pivot size, with and without pivoting
eps = 10.0 ** -np.arange(1, 21)
exact_x1 = 1.0 / (1.0 - eps)
err_naive = np.abs([naive_2x2(e)[0] for e in eps] - exact_x1)
err_piv = np.abs([pivot_2x2(e)[0] for e in eps] - exact_x1)
floor = 1e-17
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.loglog(eps, np.maximum(err_naive, floor), "o-", c=BLUE, lw=2, ms=6, label="no pivoting (pivot = ε)")
ax.loglog(eps, np.maximum(err_piv, floor), "s-", c=ORANGE, lw=2, ms=6, label="partial pivoting (swap rows)")
ax.invert_xaxis()
ax.set_xlabel("size of the first pivot ε  (smaller →)")
ax.set_ylabel("absolute error in x₁  (floored at 1e-17)")
ax.set_title("System [[ε, 1], [1, 1]] x = [1, 2]: true x₁ ≈ 1")
ax.legend(loc="center left", frameon=False)
save("02x_pivoting_error.png")

# Figure 2: time of solve (LU) vs explicit inverse
rng = np.random.default_rng(0)
ns = [100, 200, 400, 800, 1600]
t_solve, t_inv = [], []
for n in ns:
    A = rng.standard_normal((n, n)); b = rng.standard_normal(n)
    reps = 20 if n <= 400 else 5
    np.linalg.solve(A, b); np.linalg.inv(A)  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        np.linalg.solve(A, b)
    t_solve.append((time.perf_counter() - t0) / reps * 1e3)
    t0 = time.perf_counter()
    for _ in range(reps):
        np.linalg.inv(A) @ b
    t_inv.append((time.perf_counter() - t0) / reps * 1e3)
for n, s, i in zip(ns, t_solve, t_inv):
    print(f"n={n:5d}  solve {s:8.2f} ms   inv@b {i:8.2f} ms   ratio {i / s:4.1f}")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.loglog(ns, t_solve, "o-", c=BLUE, lw=2, ms=6, label="np.linalg.solve  (LU + 2 triangular solves)")
ax.loglog(ns, t_inv, "s-", c=ORANGE, lw=2, ms=6, label="np.linalg.inv(A) @ b")
ax.set_xlabel("matrix size n")
ax.set_ylabel("time per solve (ms)")
ax.set_title("Solving Ax = b: elimination vs explicit inverse")
ax.legend(loc="upper left", frameon=False)
save("02x_solve_vs_inverse.png")
