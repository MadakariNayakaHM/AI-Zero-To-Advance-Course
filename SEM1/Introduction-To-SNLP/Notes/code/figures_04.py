"""Figures for Note 04 (syntactic parsing). Writes ../images/04_*.png.

Run:  python3 figures_04.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
from nltk import Tree

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, RED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#c0392b"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})


def draw_tree(ax, t, probs=None, title=""):
    """Leaves evenly spaced along the bottom; each parent sits above the middle of its children."""
    leaves = t.leaves()
    depth = t.height()
    pos = {}

    def place(node, path, level):
        if isinstance(node, str):
            k = len([p for p in pos if pos[p][2] == "leaf"])
            pos[path] = (k, depth - level + 0.35, "leaf")   # just under its POS tag
            return k
        xs = [place(ch, path + (i,), level + 1) for i, ch in enumerate(node)]
        x = sum(xs) / len(xs)
        pos[path] = (x, depth - level, "node")
        return x

    place(t, (), 0)
    for path, (x, yv, kind) in pos.items():
        if kind == "leaf":
            ax.text(x, yv, t[path], ha="center", va="top", fontsize=11, style="italic", color=INK)
            continue
        node = t[path]
        lab = node.label()
        ax.text(x, yv, lab, ha="center", va="center", fontsize=11, weight="bold", color=BLUE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
        for i, ch in enumerate(node):
            cx, cy, _ = pos[path + (i,)]
            ax.plot([x, cx], [yv - 0.22, cy + (0.05 if isinstance(ch, str) else 0.22)], color=MUTED, lw=1)
        if probs is not None:
            p = probs(node)
            if p is not None:
                ax.text(x + 0.12, yv + 0.22, f"{p}", fontsize=8.5, color=ORANGE)
    ax.set_xlim(-0.6, len(leaves) - 0.4)
    ax.set_ylim(-0.6, depth + 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=11.5)


RULES = {("S", ("NP", "VP")): 1.0, ("VP", ("V", "NP")): 0.6, ("VP", ("V", "NP", "PP")): 0.4,
         ("NP", ("NP", "NP")): 0.1, ("NP", ("NP", "PP")): 0.2, ("NP", ("N",)): 0.7, ("PP", ("P", "NP")): 1.0,
         ("N", ("people",)): 0.5, ("N", ("fish",)): 0.2, ("N", ("tanks",)): 0.2, ("N", ("rods",)): 0.1,
         ("V", ("people",)): 0.1, ("V", ("fish",)): 0.6, ("V", ("tanks",)): 0.3, ("P", ("with",)): 1.0}


def rule_prob(node):
    rhs = tuple(c if isinstance(c, str) else c.label() for c in node)
    return RULES.get((node.label(), rhs))


# 1. The two PCFG trees for "people fish tanks with rods"
t1 = Tree.fromstring("(S (NP (N people)) (VP (V fish) (NP (N tanks)) (PP (P with) (NP (N rods)))))")
t2 = Tree.fromstring("(S (NP (N people)) (VP (V fish) (NP (NP (N tanks)) (PP (P with) (NP (N rods))))))")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
draw_tree(axes[0], t1, rule_prob, "t1: verb attachment (VP → V NP PP)\nP(t1) = 0.0008232")
draw_tree(axes[1], t2, rule_prob, "t2: noun attachment (NP → NP PP)\nP(t2) = 0.00024696")
fig.suptitle("Orange numbers = probability of the rule used at that node; P(tree) = their product.  "
             "P(s) = P(t1) + P(t2) = 0.00107016", fontsize=10.5, color=MUTED, y=0.05)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(OUT, "04_pcfg_two_trees.png"), dpi=150)
plt.close(fig)

# 2. The final CKY chart for "fish people fish tanks"
words = ["fish", "people", "fish", "tanks"]
chart = {
    (0, 1): ["N 0.2", "V 0.6", "NP→N 0.14", "VP→V 0.06", "S→VP 0.006"],
    (1, 2): ["N 0.5", "V 0.1", "NP→N 0.35", "VP→V 0.01", "S→VP 0.001"],
    (2, 3): ["N 0.2", "V 0.6", "NP→N 0.14", "VP→V 0.06", "S→VP 0.006"],
    (3, 4): ["N 0.2", "V 0.3", "NP→N 0.14", "VP→V 0.03", "S→VP 0.003"],
    (0, 2): ["NP→NP NP 0.0049", "VP→V NP 0.105", "S→VP 0.0105"],
    (1, 3): ["NP→NP NP 0.0049", "VP→V NP 0.007", "S→NP VP 0.0189"],
    (2, 4): ["NP→NP NP 0.00196", "VP→V NP 0.042", "S→VP 0.0042"],
    (0, 3): ["NP→NP NP 0.0000686", "VP→V NP 0.00147", "S→NP VP 0.000882"],
    (1, 4): ["NP→NP NP 0.0000686", "VP→V NP 0.000098", "S→NP VP 0.01323"],
    (0, 4): ["NP→NP NP 0.0000009604", "VP→V NP 0.00002058", "S→NP VP 0.00018522"],
}
fig, ax = plt.subplots(figsize=(11, 7.2))
W, H = 2.6, 1.55
for (i, j), entries in chart.items():
    x0, y0 = (j - 1) * W, (3 - i) * H
    best = (i, j) == (0, 4)
    ax.add_patch(Rectangle((x0, y0), W, H, fc="#fdf0e9" if best else "#e8f1fb", ec=MUTED, lw=1))
    for k, e in enumerate(entries):
        hl = best and e.startswith("S→")
        ax.text(x0 + 0.1, y0 + H - 0.25 - k * 0.27, e, fontsize=9, va="center",
                color=ORANGE if hl else INK, weight="bold" if hl else "normal")
    ax.text(x0 + W - 0.08, y0 + 0.1, f"[{i},{j}]", fontsize=8, color=MUTED, ha="right")
for j, w in enumerate(words, 1):
    ax.text((j - 1) * W + W / 2, 4 * H + 0.15, f"{w}", ha="center", fontsize=12, weight="bold")
    ax.text((j - 1) * W + W / 2, 4 * H + 0.5, f"word {j}", ha="center", fontsize=8.5, color=MUTED)
ax.set_xlim(-0.2, 4 * W + 0.2)
ax.set_ylim(-0.2, 4 * H + 0.9)
ax.axis("off")
ax.set_title("Probabilistic CKY chart for “fish people fish tanks”: cell [i,j] covers words i+1 … j.\n"
             "Diagonal = lexical + unary rules; others = best binary split (+ unaries). "
             "Answer: top-right S = 0.00018522", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_cky_chart.png"), dpi=150)
plt.close(fig)


def draw_arcs(ax, words, heads, labels=None, highlight=(), y=0.0, title=""):
    for k, w in enumerate(words):
        ax.text(k, y, w, ha="center", va="top", fontsize=12)
    for d, h in enumerate(heads):
        if h is None:
            continue
        a, b = sorted((h, d))
        width = b - a
        col = RED if (h, d) in highlight else BLUE
        ax.add_patch(Arc(((a + b) / 2, y + 0.12), width, 0.55 * width + 0.3, theta1=0, theta2=180,
                         color=col, lw=2.2 if col == RED else 1.4))
        # arrowhead at the dependent
        ax.annotate("", xy=(d, y + 0.12), xytext=(d + (0.06 if d < h else -0.06), y + 0.32),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
        if labels:
            ax.text((a + b) / 2, y + 0.12 + (0.55 * width + 0.3) / 2 + 0.05, labels[d],
                    ha="center", fontsize=8.5, color=col,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    ax.set_xlim(-0.7, len(words) - 0.3)
    ax.set_ylim(y - 0.6, y + 0.12 + max(0.55 * (len(words) - 1) + 0.3, 1) / 2 + 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=11.5)


# 3. Dependency trees: the slide's two examples
fig, axes = plt.subplots(2, 1, figsize=(11, 7.4))
draw_arcs(axes[0], ["ROOT", "The", "boy", "put", "the", "tortoise", "on", "the", "rug"],
          [None, 2, 3, 0, 5, 3, 3, 8, 6],
          [None, "det", "nsubj", "root", "det", "dobj", "prep", "det", "pobj"],
          title="Arrows go from the head to its dependent; every word has exactly one head")
draw_arcs(axes[1], ["ROOT", "Happy", "children", "like", "to", "play", "with", "their", "friends", "."],
          [None, 2, 3, 0, 5, 3, 5, 8, 6, 3],
          [None, "amod", "nsubj", "root", "aux", "xcomp", "prep", "poss", "pobj", "punct"],
          title="The arc-eager example from the slide (9 words → 9 arcs)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_dependency_trees.png"), dpi=150)
plt.close(fig)

# 4. Projectivity
fig, ax = plt.subplots(figsize=(11, 4.2))
w = ["ROOT", "Who", "did", "Bill", "buy", "the", "coffee", "from", "yesterday", "?"]
h = [None, 7, 4, 4, 0, 6, 4, 4, 4, 4]
draw_arcs(ax, w, h, highlight={(7, 1)},
          title="Non-projective: the red arc from → Who crosses buy → yesterday, buy → ? and ROOT → buy")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_projectivity.png"), dpi=150)
plt.close(fig)
print("wrote 04_pcfg_two_trees.png, 04_cky_chart.png, 04_dependency_trees.png, 04_projectivity.png")
