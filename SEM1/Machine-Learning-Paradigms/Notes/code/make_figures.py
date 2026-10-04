"""Generates every figure used in the MLP notes.

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

# Slide dataset: word count (X) -> votes (Y)
X = np.array([7, 8, 4, 7, 5], dtype=float)
Y = np.array([3, 15, 5, 20, 4], dtype=float)
m_ols = ((X - X.mean()) * (Y - Y.mean())).sum() / ((X - X.mean()) ** 2).sum()
c_ols = Y.mean() - m_ols * X.mean()


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


# 1. Mean baseline vs OLS line vs the line printed on the slide
xs = np.linspace(3, 9, 50)
plt.figure(figsize=(6.5, 4))
plt.scatter(X, Y, s=60, zorder=3, label="data (word count, votes)")
plt.axhline(Y.mean(), ls="--", c="gray", label=f"Case 1: predict mean = {Y.mean():.1f}")
plt.plot(xs, m_ols * xs + c_ols, c="C2", lw=2, label=f"OLS: y = {m_ols:.2f}x {c_ols:+.2f}  (correct)")
plt.plot(xs, 4.41 * xs - 18, c="C3", ls=":", lw=2, label="slide: y = 4.41x - 18.00")
for xi, yi in zip(X, Y):
    plt.plot([xi, xi], [yi, m_ols * xi + c_ols], c="C2", alpha=0.4)
plt.xlabel("Word count (X)")
plt.ylabel("Votes (Y)")
plt.title("Baseline (mean) vs best-fit line")
plt.legend(fontsize=8)
save("01_mean_vs_ols.png")

# 2. MSE as a function of slope (intercept fixed at its optimum) -> convex parabola
ms = np.linspace(-3, 8, 200)
mse_m = [np.mean((Y - (m * X + c_ols)) ** 2) for m in ms]
plt.figure(figsize=(6, 3.8))
plt.plot(ms, mse_m)
plt.scatter([m_ols], [np.mean((Y - (m_ols * X + c_ols)) ** 2)], c="C3", zorder=3, label=f"minimum at m = {m_ols:.2f}")
plt.xlabel("slope m  (c fixed)")
plt.ylabel("MSE")
plt.title("MSE vs slope: a convex bowl (single global minimum)")
plt.legend()
save("02_mse_vs_slope.png")

# 3. MSE contour over (w0, w1) with a gradient-descent path
def mse(w0, w1):
    return np.mean((w0 + w1 * X[:, None, None] - Y[:, None, None]) ** 2, axis=0)

W0, W1 = np.meshgrid(np.linspace(-30, 15, 200), np.linspace(-2, 7, 200))
Z = mse(W0, W1)
w0, w1, alpha, path = 0.0, 0.0, 0.02, [(0.0, 0.0)]
for _ in range(3000):
    err = w0 + w1 * X - Y
    w0 -= alpha * err.mean()
    w1 -= alpha * (err * X).mean()
    path.append((w0, w1))
path = np.array(path)
plt.figure(figsize=(6.5, 4.5))
cs = plt.contour(W0, W1, Z, levels=np.logspace(1, 3.2, 18), cmap="viridis")
plt.plot(path[:, 0], path[:, 1], "r.-", ms=2, lw=1, label="GD path (alpha = 0.02)")
plt.scatter([c_ols], [m_ols], marker="*", s=200, c="k", zorder=3, label="OLS optimum")
plt.xlabel("w0 (intercept)")
plt.ylabel("w1 (slope)")
plt.title("MSE contours + gradient descent path")
plt.legend(fontsize=8)
save("03_mse_contour_gd_path.png")

# 4. Effect of learning rate (on a bigger synthetic dataset, standardised x)
rng = np.random.default_rng(0)
xb = rng.uniform(0, 10, 200)
yb = 3 * xb + 5 + rng.normal(0, 2, 200)
xs_ = (xb - xb.mean()) / xb.std()


def run_gd(alpha, iters=60):
    w0 = w1 = 0.0
    losses = []
    for _ in range(iters):
        err = w0 + w1 * xs_ - yb
        losses.append(np.mean(err ** 2) / 2)
        w0 -= alpha * err.mean()
        w1 -= alpha * (err * xs_).mean()
    return losses

plt.figure(figsize=(6.5, 4))
for a, lab in [(0.005, "too small (0.005): slow"), (0.1, "moderate (0.1): smooth"), (1.0, "large (1.0): fast here"), (2.05, "too large (2.05): diverges")]:
    plt.plot(run_gd(a), label=lab)
plt.yscale("log")
plt.ylim(1, 1e4)
plt.xlabel("iteration")
plt.ylabel("cost C(w)  (log scale)")
plt.title("Learning rate decides speed vs stability")
plt.legend(fontsize=8)
save("04_learning_rate.png")

# 5. Batch vs Stochastic vs Mini-batch GD


def train(batch, epochs=30, alpha=0.05, seed=1):
    r = np.random.default_rng(seed)
    w0 = w1 = 0.0
    hist = []
    n = len(xs_)
    for _ in range(epochs):
        idx = r.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            err = w0 + w1 * xs_[b] - yb[b]
            w0 -= alpha * err.mean()
            w1 -= alpha * (err * xs_[b]).mean()
            hist.append(np.mean((w0 + w1 * xs_ - yb) ** 2) / 2)
    return np.array(hist), n // batch + (n % batch > 0)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for batch, lab in [(200, "Batch GD (all 200)"), (1, "Stochastic GD (1)"), (16, "Mini-batch GD (16)")]:
    h, steps = train(batch)
    ax[0].plot(np.arange(len(h)) / steps, h, label=lab, alpha=0.85)
    ax[1].plot(h[:200], "o-" if batch == 200 else "-", ms=3, label=lab, alpha=0.85)
ax[0].set(xlabel="epoch", ylabel="cost (full data)", yscale="log", title="Cost vs epochs")
ax[1].set(xlabel="parameter update #", ylabel="cost", yscale="log", title="Cost vs updates (batch GD makes only 30)")
ax[0].legend(fontsize=8)
save("05_gd_variants.png")

# 6. Polynomial regression: under-fit / good fit / over-fit
r = np.random.default_rng(3)
xp = np.sort(r.uniform(-3, 3, 20))
yp = 0.5 * xp ** 3 - xp ** 2 + 2 + r.normal(0, 2, 20)
grid = np.linspace(-3, 3, 300)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for a, deg, t in zip(ax, [1, 3, 15], ["degree 1: under-fit (high bias)", "degree 3: good fit", "degree 15: over-fit (high variance)"]):
    coef = np.polyfit(xp, yp, deg)
    a.scatter(xp, yp, s=20)
    a.plot(grid, np.polyval(coef, grid), c="C3")
    a.set_ylim(-25, 15)
    a.set_title(t)
save("06_polynomial_fit.png")

print(f"OLS on slide data: m = {m_ols:.4f}, c = {c_ols:.4f}")
