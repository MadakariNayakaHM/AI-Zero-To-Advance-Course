"""Figures for Note 02 (Neural Networks Fundamentals). Run from anywhere:
    python3 figures_02.py
Writes images/02x_*.png next to the Notes folder."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images")
os.makedirs(OUT, exist_ok=True)
sig = lambda z: 1 / (1 + np.exp(-z))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)
    print("wrote", name)


# 1. sigmoid derivative and the 0.25^k chain
z = np.linspace(-8, 8, 400)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(z, sig(z), label="σ(z)")
ax[0].plot(z, sig(z) * (1 - sig(z)), label="σ'(z) = σ(1−σ)")
ax[0].plot(z, 1 - np.tanh(z) ** 2, "--", label="tanh'(z)")
ax[0].plot(z, (z > 0).astype(float), ":", label="ReLU'(z)")
ax[0].axhline(0.25, color="gray", lw=0.6)
ax[0].set_title("activation derivatives (σ' peaks at 0.25)")
ax[0].legend(fontsize=8)
k = np.arange(1, 11)
ax[1].semilogy(k, 0.25 ** k, "o-", label="sigmoid best case 0.25ᵏ")
ax[1].semilogy(k, 0.10499 ** k, "s-", label="sigmoid at |z| = 2: 0.105ᵏ")
ax[1].semilogy(k, np.ones_like(k, dtype=float), "^-", label="ReLU active unit: 1ᵏ")
ax[1].set_xlabel("number of layers k")
ax[1].set_title("product of activation derivatives")
ax[1].legend(fontsize=8)
save(fig, "02x_sigmoid_chain.png")

# shared small MLP (column convention, W is out × in)
digits = load_digits()
X = (digits.data / 16.0).T[:, :256]
Y = np.eye(10)[:, digits.target[:256]]
ACT = {"sigmoid": (sig, lambda a, h: h * (1 - h)),
       "tanh": (np.tanh, lambda a, h: 1 - h ** 2),
       "relu": (lambda a: np.maximum(0, a), lambda a, h: (a > 0).astype(float))}


def init(sizes, scheme, rng):
    Ws = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        s = {"small": 0.01, "large": 1.0, "xavier": np.sqrt(2 / (n_in + n_out)), "he": np.sqrt(2 / n_in)}[scheme]
        Ws.append(rng.standard_normal((n_out, n_in)) * s)
    return Ws


def grads(sizes, act, scheme, seed=0):
    g, dg = ACT[act]
    Ws = init(sizes, scheme, np.random.default_rng(seed))
    hs, As = [X], []
    for l, W in enumerate(Ws):
        A = W @ hs[-1]
        As.append(A)
        if l == len(Ws) - 1:
            E = np.exp(A - A.max(0)); hs.append(E / E.sum(0))
        else:
            hs.append(g(A))
    d = (hs[-1] - Y) / X.shape[1]
    norms = []
    for l in range(len(Ws) - 1, -1, -1):
        norms.append(np.linalg.norm(d @ hs[l].T))
        if l > 0:
            d = (Ws[l].T @ d) * dg(As[l - 1], hs[l])
    return norms[::-1], [h.std() for h in hs[1:-1]]


# 2. vanishing gradients: per-layer gradient norm at init, 10 hidden layers
sizes = [64] + [64] * 10 + [10]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for act, sch in [("sigmoid", "xavier"), ("tanh", "xavier"), ("relu", "he")]:
    n, _ = grads(sizes, act, sch)
    ax.semilogy(range(1, len(n) + 1), n, "o-", label=f"{act} + {sch}")
ax.set_xlabel("layer (1 = nearest the input)")
ax.set_ylabel("‖∂L/∂W‖")
ax.set_title("gradient norm per layer at initialisation (10 hidden layers)")
ax.legend()
save(fig, "02x_vanishing_gradients.png")

# 3. activation scale per layer for different initialisations (ReLU, 6 hidden layers)
sizes = [64] + [64] * 6 + [10]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for sch in ["small", "large", "xavier", "he"]:
    _, s = grads(sizes, "relu", sch)
    ax.semilogy(range(1, 7), s, "o-", label=sch)
ax.set_xlabel("hidden layer")
ax.set_ylabel("std of activations")
ax.set_title("ReLU network: activation scale vs initialisation")
ax.legend()
save(fig, "02x_init_scale.png")

# 4. universal approximation with ReLU hats, and the tent map
f = lambda x: np.sin(2 * np.pi * x) + 0.5 * x
xs = np.linspace(0, 1, 1001)


def relu_interp(N):
    knots = np.linspace(0, 1, N + 1); vals = f(knots)
    slopes = np.diff(vals) / np.diff(knots)
    coef = np.r_[slopes[0], np.diff(slopes)]
    return vals[0] + sum(c * np.maximum(0, xs - kk) for c, kk in zip(coef, knots[:-1]))


tent = lambda x: 2 * np.maximum(0, x) - 4 * np.maximum(0, x - 0.5)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(xs, f(xs), "k", lw=2, label="target f(x)")
for N in [4, 8, 16]:
    ax[0].plot(xs, relu_interp(N), label=f"{N} hidden ReLUs")
ax[0].set_title("one hidden layer: sum of ReLU 'bends'")
ax[0].legend(fontsize=8)
v = xs.copy()
for kk in range(1, 4):
    v = tent(v)
    ax[1].plot(xs, v + 1.2 * (kk - 1), label=f"tent composed {kk}× ({2 ** kk} pieces)")
ax[1].set_title("depth: k compositions give 2ᵏ linear pieces")
ax[1].set_ylim(-0.1, 4.9)
ax[1].legend(fontsize=8, loc="upper center", ncol=2)
save(fig, "02x_relu_bumps_depth.png")
