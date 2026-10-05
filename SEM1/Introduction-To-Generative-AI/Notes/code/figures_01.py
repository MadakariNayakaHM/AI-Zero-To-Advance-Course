"""Figures for Note 01 (AI/ML/DL foundations, deep-dive sections).

Run from this folder:  python figures_01.py
Figures are written to ../images/01x_*.png
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


# 1. Loss functions: regression losses vs residual, classification losses vs margin
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
r = np.linspace(-3, 3, 601)
ax[0].plot(r, r**2, label="squared  r²")
ax[0].plot(r, np.abs(r), label="absolute  |r|")
ax[0].set_xlabel("residual r = h(x) − y"); ax[0].set_ylabel("loss"); ax[0].set_ylim(0, 4)
ax[0].set_title("Regression losses"); ax[0].legend()
m = np.linspace(-3, 3, 601)
ax[1].plot(m, (m <= 0).astype(float), label="0/1  1[m ≤ 0]", lw=2)
ax[1].plot(m, np.log2(1 + np.exp(-m)), label="log loss (base 2)")
ax[1].plot(m, np.maximum(0, 1 - m), label="hinge (SVM)")
ax[1].plot(m, (1 - m)**2, label="squared (1 − m)²", alpha=0.7)
ax[1].set_xlabel("margin m = y·f(x),  y ∈ {−1,+1}"); ax[1].set_ylim(0, 4)
ax[1].set_title("Classification losses (surrogates upper-bound 0/1)"); ax[1].legend(fontsize=8)
save("01x_loss_functions.png")

# 2. Constant prediction for d = {1,2,3,4,20}: MSE minimised at mean, MAE at median
d = np.array([1, 2, 3, 4, 20.])
c = np.linspace(0, 12, 1201)
mse = ((d[None, :] - c[:, None])**2).mean(1)
mae = np.abs(d[None, :] - c[:, None]).mean(1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(c, mse); ax[0].axvline(d.mean(), ls="--", c="k")
ax[0].set_title(f"MSE(c): min {mse.min():.0f} at mean = {d.mean():.0f}")
ax[1].plot(c, mae, c="C1"); ax[1].axvline(np.median(d), ls="--", c="k")
ax[1].set_title(f"MAE(c): min {mae.min():.1f} at median = {np.median(d):.0f}")
for a in ax:
    a.set_xlabel("constant prediction c")
save("01x_mean_vs_median.png")

# 3. Hoeffding / union bound: required n vs epsilon for several |H|
eps = np.linspace(0.01, 0.2, 200)
delta = 0.05
fig, ax = plt.subplots(figsize=(6.5, 4))
for H in [1, 1e3, 1e6, 1e9]:
    n = (math.log(H) + math.log(2 / delta)) / (2 * eps**2)
    ax.plot(eps, n, label=f"|H| = {H:.0e}" if H > 1 else "single h")
ax.set_yscale("log"); ax.set_xlabel("accuracy ε"); ax.set_ylabel("samples n needed (δ = 0.05)")
ax.set_title("n ≥ (ln|H| + ln(2/δ)) / (2ε²)"); ax.legend(); ax.grid(alpha=0.3, which="both")
save("01x_hoeffding_sample_size.png")

# 4. Bayes-optimal classifier for two 1-D Gaussians (equal and unequal priors)
xs = np.linspace(-4, 6, 1001)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, p1 in zip(ax, [0.5, 0.8]):
    p0 = 1 - p1
    a.plot(xs, p0 * norm.pdf(xs, 0, 1), label="P(y=0) p(x|y=0),  N(0,1)")
    a.plot(xs, p1 * norm.pdf(xs, 2, 1), label="P(y=1) p(x|y=1),  N(2,1)")
    t = 1 - math.log(p1 / p0) / 2
    err = p0 * (1 - norm.cdf(t)) + p1 * norm.cdf(t - 2)
    a.axvline(t, c="k", ls="--")
    a.fill_between(xs, 0, np.minimum(p0 * norm.pdf(xs, 0, 1), p1 * norm.pdf(xs, 2, 1)), color="grey", alpha=0.4,
                   label="Bayes error (shaded)")
    a.set_title(f"P(y=1)={p1}: threshold {t:.3f}, Bayes error {err:.4f}")
    a.set_xlabel("x"); a.legend(fontsize=7)
save("01x_bayes_optimal.png")

# 5. Bias-variance trade-off: polynomial degree on y = sin(2πx) + N(0, 0.3²), n = 30,
#    bias² and variance averaged over a grid of test points in [0, 1]
rng = np.random.default_rng(1)
sigma = 0.3
xg = np.linspace(0, 1, 101)
fg = np.sin(2 * np.pi * xg)
degs = list(range(0, 9))
b2s, vs = [], []
for deg in degs:
    preds = []
    for _ in range(400):
        x = rng.uniform(0, 1, 30)
        y = np.sin(2 * np.pi * x) + rng.normal(0, sigma, 30)
        coef = np.polynomial.polynomial.polyfit(x - 0.5, y, deg)
        preds.append(np.polynomial.polynomial.polyval(xg - 0.5, coef))
    preds = np.array(preds)
    b2s.append(np.mean((preds.mean(0) - fg)**2)); vs.append(np.mean(preds.var(0)))
b2s, vs = np.array(b2s), np.array(vs)
for deg, b, v in zip(degs, b2s, vs):
    print(f"degree {deg:>2}: bias^2={b:.4f} variance={v:.4f} total={b + v + sigma**2:.4f}")
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(degs, b2s, "o-", label="bias²")
ax.plot(degs, vs, "s-", label="variance")
ax.plot(degs, b2s + vs + sigma**2, "^-", label="bias² + variance + noise")
ax.axhline(sigma**2, ls=":", c="k", label="noise σ² = 0.09")
ax.set_yscale("log"); ax.set_xlabel("polynomial degree (model complexity)")
ax.set_ylabel("expected squared error (avg over x)"); ax.set_title("Bias–variance trade-off (n = 30)")
ax.legend(fontsize=8)
save("01x_bias_variance.png")
