"""Figures for Note 02 (lexical processing, part 1). Writes ../images/02x_*.png.

Run:  python3 figures_02.py   (needs nltk 'gutenberg' data)
"""
import os
import re
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})

TEXTS = [("carroll-alice.txt", "Alice", BLUE), ("austen-emma.txt", "Emma", ORANGE),
         ("melville-moby_dick.txt", "Moby Dick", AQUA)]


def tokens(fid):
    return re.findall(r"[a-z]+", gutenberg.raw(fid).lower())


# 1. Zipf: rank-frequency on log-log axes
fig, ax = plt.subplots(figsize=(7, 4.6))
for fid, name, col in TEXTS:
    f = np.array(sorted(Counter(tokens(fid)).values(), reverse=True))
    r = np.arange(1, len(f) + 1)
    s, b = np.polyfit(np.log10(r[:1000]), np.log10(f[:1000]), 1)
    ax.loglog(r, f, color=col, lw=2, label=f"{name} (fitted slope {s:.2f})")
r = np.logspace(0, 4.3, 50)
ax.loglog(r, 15000 / r, color=MUTED, lw=1.2, ls="--", label="ideal Zipf, slope -1")
ax.set_xlabel("rank r (log scale)")
ax.set_ylabel("frequency f (log scale)")
ax.set_title("Zipf's law: log f is roughly linear in log r", color=INK)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25, which="major")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02x_zipf_loglog.png"), dpi=130)
plt.close(fig)

# 2. Heaps: vocabulary growth
fig, ax = plt.subplots(figsize=(7, 4.6))
for fid, name, col in TEXTS:
    t = tokens(fid)
    seen, Ns, Vs = set(), [], []
    for i, w in enumerate(t, 1):
        seen.add(w)
        if i % 1000 == 0:
            Ns.append(i)
            Vs.append(len(seen))
    beta, lk = np.polyfit(np.log(Ns), np.log(Vs), 1)
    ax.plot(np.array(Ns) / 1000, Vs, color=col, lw=2,
            label=f"{name}: V ≈ {np.exp(lk):.1f}·N^{beta:.2f}")
ax.set_xlabel("tokens read N (thousands)")
ax.set_ylabel("distinct types V")
ax.set_title("Heaps' law: the vocabulary never stops growing", color=INK)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02x_heaps.png"), dpi=130)
plt.close(fig)

# 3. Levenshtein DP table with the backtrace path (kitten -> sitting)
a, b = "kitten", "sitting"
D = np.zeros((len(a) + 1, len(b) + 1), int)
D[:, 0] = range(len(a) + 1)
D[0, :] = range(len(b) + 1)
for i in range(1, len(a) + 1):
    for j in range(1, len(b) + 1):
        D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
path, i, j = [], len(a), len(b)
while i > 0 or j > 0:
    path.append((i, j))
    if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
        i, j = i - 1, j - 1
    elif i > 0 and D[i, j] == D[i - 1, j] + 1:
        i -= 1
    else:
        j -= 1
path.append((0, 0))
fig, ax = plt.subplots(figsize=(6.4, 5.4))
ax.imshow(D, cmap="Blues", vmin=-2, vmax=10)
for (p, q) in path:
    ax.add_patch(plt.Rectangle((q - 0.5, p - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.5))
for p in range(D.shape[0]):
    for q in range(D.shape[1]):
        ax.text(q, p, D[p, q], ha="center", va="center", color=INK, fontsize=11)
ax.set_xticks(range(len(b) + 1), ["#"] + list(b))
ax.set_yticks(range(len(a) + 1), ["#"] + list(a))
ax.xaxis.tick_top()
ax.set_title("Levenshtein DP: kitten → sitting = 3 (orange = backtrace)", color=INK, pad=28)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02x_levenshtein_dp.png"), dpi=130)
plt.close(fig)


# 4. BPE vocabulary-size trade-off: train on Alice, test on a slice of Moby Dick
def get_stats(vocab):
    pairs = Counter()
    for word, f in vocab.items():
        for x, y in zip(word, word[1:]):
            pairs[(x, y)] += f
    return pairs


def merge_word(word, pair):
    out, k = [], 0
    while k < len(word):
        if k < len(word) - 1 and (word[k], word[k + 1]) == pair:
            out.append(word[k] + word[k + 1])
            k += 2
        else:
            out.append(word[k])
            k += 1
    return tuple(out)


train_counts = Counter(tokens("carroll-alice.txt"))
vocab = {tuple(w) + ("_",): f for w, f in train_counts.items()}
test_counts = Counter(tokens("melville-moby_dick.txt")[:50000])
test_vocab = {tuple(w) + ("_",): f for w, f in test_counts.items()}
n_test = sum(test_counts.values())
base = len({s for w in vocab for s in w})
checkpoints = [0, 50, 100, 200, 400, 700, 1000, 1500, 2000]
Ks, sizes, tpw = [], [], []
for k in range(checkpoints[-1] + 1):
    if k in checkpoints:
        Ks.append(k)
        sizes.append(base + k)
        tpw.append(sum(len(w) * f for w, f in test_vocab.items()) / n_test)
    st = get_stats(vocab)
    if not st:
        break
    best = max(st, key=st.get)
    vocab = {merge_word(w, best): f for w, f in vocab.items()}
    test_vocab = {merge_word(w, best): f for w, f in test_vocab.items()}
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(sizes, tpw, color=BLUE, lw=2, marker="o", ms=8)
for s, t in list(zip(sizes, tpw))[::2]:
    ax.annotate(f"{t:.2f}", (s, t), textcoords="offset points", xytext=(6, 6), color=INK, fontsize=9)
ax.set_xlabel("vocabulary size (27 base symbols + number of merges)")
ax.set_ylabel("tokens per word on unseen text")
ax.set_title("BPE trained on Alice, tested on Moby Dick: bigger vocab, shorter sequences",
             color=INK, fontsize=10.5)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02x_bpe_vocab_tradeoff.png"), dpi=130)
plt.close(fig)
print("base symbols", base)
for k, s, t in zip(Ks, sizes, tpw):
    print(f"merges={k:5d}  vocab={s:5d}  tokens/word={t:.3f}")
print("saved figures to", os.path.abspath(OUT))
