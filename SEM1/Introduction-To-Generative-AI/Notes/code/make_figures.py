"""Generates every figure used in the Gen AI notes.

Run from this folder:  python make_figures.py
Figures are written to ../images/
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


# 1. Nested Venn: AI ⊃ ML ⊃ DL ⊃ Gen AI
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for (w, h, y, c, label, ty) in [
    (6.0, 4.2, 0.0, "#c6dbef", "Artificial Intelligence\nrule-based systems, search, planning, chess engines", 1.6),
    (4.6, 3.0, -0.5, "#9ecae1", "Machine Learning\nlinear/logistic regression, trees, SVM, k-means", 0.55),
    (3.2, 1.9, -1.0, "#6baed6", "Deep Learning\nCNNs, RNNs, Transformers", -0.35),
    (1.8, 0.8, -1.45, "#3182bd", "Generative AI\nLLMs, diffusion", -1.45),
]:
    ax.add_patch(Ellipse((0, y), w, h, fc=c, ec="k", lw=1))
    ax.text(0, ty, label, ha="center", va="center", fontsize=8.5 if "Generative" not in label else 8,
            color="white" if "Generative" in label else "black")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.2, 2.2); ax.axis("off")
ax.set_title("DL ⊂ ML ⊂ AI  (and Gen AI ⊂ DL)")
save("01_ai_ml_dl_venn.png")

# 2. XOR is not linearly separable
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a in ax:
    a.scatter(X[y == 0, 0], X[y == 0, 1], s=200, c="C0", marker="o", label="output 0", zorder=3)
    a.scatter(X[y == 1, 0], X[y == 1, 1], s=200, c="C3", marker="X", label="output 1", zorder=3)
    for (p, q), t in zip(X, y):
        a.text(p + 0.06, q + 0.06, f"({p},{q})→{t}")
    a.set_xlim(-0.4, 1.5); a.set_ylim(-0.4, 1.5); a.set_aspect("equal"); a.set_xlabel("x₁"); a.set_ylabel("x₂"); a.grid(alpha=0.3)
xs = np.linspace(-0.4, 1.5, 10)
for k, ls in [(0.5, "--"), (1.0, ":"), (1.5, "-.")]:
    ax[0].plot(xs, k - xs, ls, c="gray")
ax[0].set_title("XOR: no single straight line\nseparates ✕ from ●")
ax[0].legend(loc="upper right", fontsize=8)
ax[1].plot(xs, 0.5 - xs, c="g", lw=2, label="h₁: x₁+x₂ ≥ 0.5 (OR)")
ax[1].plot(xs, 1.5 - xs, c="purple", lw=2, label="h₂: x₁+x₂ ≥ 1.5 (AND)")
ax[1].fill_between(xs, 0.5 - xs, 1.5 - xs, color="C3", alpha=0.12)
ax[1].set_title("Two hidden neurons = two lines;\noutput = OR AND (NOT AND)")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7, ncol=2)
save("02_xor.png")

# 3. Activation functions and their derivatives
z = np.linspace(-5, 5, 400)
sig = 1 / (1 + np.exp(-z))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(z, sig, label="sigmoid σ(z) ∈ (0,1)")
ax[0].plot(z, np.tanh(z), label="tanh(z) ∈ (−1,1)")
ax[0].plot(z, np.maximum(0, z), label="ReLU(z) ∈ [0,∞)")
ax[0].set_ylim(-1.5, 3); ax[0].set_title("Activation functions g(z)")
ax[1].plot(z, sig * (1 - sig), label="σ' = σ(1−σ)  (max 0.25)")
ax[1].plot(z, 1 - np.tanh(z) ** 2, label="tanh' = 1 − tanh²  (max 1)")
ax[1].plot(z, (z > 0).astype(float), label="ReLU' = 1 if z>0 else 0")
ax[1].set_title("Derivatives (used in backpropagation)")
for a in ax:
    a.axhline(0, c="k", lw=0.6); a.axvline(0, c="k", lw=0.6); a.grid(alpha=0.3); a.legend(fontsize=8); a.set_xlabel("z")
save("03_activations.png")

# 4. Softmax: turning scores into probabilities (and temperature)
logits = np.array([2.0, 1.0, 0.1])
labels = ["cat", "dog", "horse"]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
for a, T in zip(ax, [0.5, 1.0, 3.0]):
    p = np.exp(logits / T) / np.exp(logits / T).sum()
    a.bar(labels, p, color=["C0", "C1", "C2"])
    for i, v in enumerate(p):
        a.text(i, v + 0.02, f"{v:.2f}", ha="center")
    a.set_title(f"softmax(z / T), T = {T}" + ("  (standard)" if T == 1 else ""))
    a.set_ylim(0, 1.05)
ax[0].set_ylabel("probability")
fig.suptitle("Logits z = [2.0, 1.0, 0.1] → probabilities summing to 1  (low T = confident, high T = flatter: same 'temperature' as in ChatGPT)", fontsize=9)
save("04_softmax_temperature.png")

# 5. Why nonlinearity: a trained 2-layer net on XOR-like data vs logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
rng = np.random.default_rng(0)
Xd = rng.uniform(-1, 1, (400, 2))
yd = ((Xd[:, 0] > 0) ^ (Xd[:, 1] > 0)).astype(int)
gx, gy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
G = np.c_[gx.ravel(), gy.ravel()]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
models = [("Logistic regression (linear)", LogisticRegression()),
          ("MLP, 8 hidden, identity activation", MLPClassifier((8,), activation="identity", max_iter=3000, random_state=0)),
          ("MLP, 8 hidden, ReLU activation", MLPClassifier((8,), activation="relu", max_iter=3000, random_state=0))]
for a, (name, m) in zip(ax, models):
    m.fit(Xd, yd)
    a.contourf(gx, gy, m.predict(G).reshape(gx.shape), alpha=0.3, cmap="coolwarm")
    a.scatter(Xd[:, 0], Xd[:, 1], c=yd, cmap="coolwarm", s=8)
    a.set_title(f"{name}\naccuracy = {m.score(Xd, yd):.2f}", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
save("05_linear_vs_nonlinear.png")
