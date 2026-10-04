"""Figures for 01-Introduction-to-ML-Paradigms.md (Sections 18-21).

Run from this folder:  python figures_01.py
Writes ../images/01_paradigms_same_data.png, 01_gridworld_values.png, 01_drift_psi_ks.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print("saved", name)


# 1. Same data, different paradigms (identical setup to the notebook, Part A)
X, y = make_moons(n_samples=300, noise=0.1, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
rng = np.random.default_rng(0)
few = np.concatenate([rng.choice(np.where(y_tr == c)[0], 3, replace=False) for c in (0, 1)])
sup_few = SVC(kernel="rbf", gamma=2.0).fit(X_tr[few], y_tr[few])
y_semi = np.full_like(y_tr, -1)
y_semi[few] = y_tr[few]
lp = LabelPropagation(kernel="knn", n_neighbors=10, max_iter=2000).fit(X_tr, y_semi)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=15, random_state=0).fit(X)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
ax[0].scatter(X_tr[:, 0], X_tr[:, 1], c=sup_few.predict(X_tr), cmap="coolwarm", s=10, alpha=0.6)
ax[0].scatter(X_tr[few, 0], X_tr[few, 1], c=y_tr[few], cmap="coolwarm", s=110, edgecolor="k")
ax[0].set_title("Supervised, 6 labels only (big dots)")
ax[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="coolwarm", s=10)
ax[1].set_title("Unsupervised: K-means (no labels)")
ax[2].scatter(X[:, 0], X[:, 1], c=sc.labels_, cmap="coolwarm", s=10)
ax[2].set_title("Unsupervised: spectral clustering")
ax[3].scatter(X_tr[:, 0], X_tr[:, 1], c=lp.transduction_, cmap="coolwarm", s=10)
ax[3].scatter(X_tr[few, 0], X_tr[few, 1], c=y_tr[few], cmap="coolwarm", s=110, edgecolor="k")
ax[3].set_title("Semi-supervised: label propagation")
save("01_paradigms_same_data.png")

# 2. Gridworld optimal values (value iteration) and greedy policy
GOAL, PIT, gamma = 15, 5, 0.9
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ARROWS = "^>v<"


def step(s, a):
    r, c = divmod(s, 4)
    nr, nc = min(max(r + MOVES[a][0], 0), 3), min(max(c + MOVES[a][1], 0), 3)
    s2 = nr * 4 + nc
    if s2 == GOAL:
        return s2, 10.0, True
    if s2 == PIT:
        return s2, -10.0, True
    return s2, -1.0, False


V = np.zeros(16)
for _ in range(100):
    V_new = V.copy()
    for s in range(16):
        if s in (GOAL, PIT):
            continue
        V_new[s] = max(r + gamma * (0 if d else V[s2]) for s2, r, d in (step(s, a) for a in range(4)))
    V = V_new
pol = []
for s in range(16):
    if s == GOAL:
        pol.append("G")
    elif s == PIT:
        pol.append("PIT")
    else:
        q = [r + gamma * (0 if d else V[s2]) for s2, r, d in (step(s, a) for a in range(4))]
        pol.append("".join(ARROWS[a] for a in range(4) if abs(q[a] - max(q)) < 1e-9))
plt.figure(figsize=(4.8, 4.2))
plt.imshow(V.reshape(4, 4), cmap="viridis")
for s in range(16):
    plt.text(s % 4, s // 4, f"{pol[s]}\n{V[s]:.2f}", ha="center", va="center", color="w", fontsize=10)
plt.colorbar(label="V*(s)")
plt.xticks([]); plt.yticks([]); plt.grid(False)
plt.title("4x4 gridworld: V*(s) and optimal actions\n(-1 per move, +10 goal, -10 pit, gamma = 0.9)")
save("01_gridworld_values.png")

# 3. Drift: training vs live histograms, PSI and KS
rng = np.random.default_rng(7)
train = rng.normal(0, 1, 5000)
edges = np.quantile(train, np.linspace(0, 1, 11))
edges[0], edges[-1] = -np.inf, np.inf
e_share = np.histogram(train, edges)[0] / len(train)


def psi(e, a, eps=1e-4):
    e = np.clip(e, eps, None); a = np.clip(a, eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
for axi, shift in zip(ax, [0.0, 0.25, 1.0]):
    live = rng.normal(shift, 1, 2000)
    a_share = np.histogram(live, edges)[0] / len(live)
    ks = stats.ks_2samp(train, live)
    axi.hist(train, bins=40, range=(-4, 5), density=True, alpha=0.5, label="training")
    axi.hist(live, bins=40, range=(-4, 5), density=True, alpha=0.5, label="live")
    axi.set_title(f"mean shift {shift}: PSI = {psi(e_share, a_share):.3f}, KS D = {ks.statistic:.3f}")
    axi.legend()
save("01_drift_psi_ks.png")
