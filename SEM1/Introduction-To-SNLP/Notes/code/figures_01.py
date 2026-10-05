"""Figures for Note 01 (information layers). Writes ../images/01_*.png."""
import math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import lfilter, spectrogram

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})

fs, F0 = 10000, 120
def synth(formants, dur=0.4, bws=(60, 90, 120)):
    n = int(fs * dur); y = np.zeros(n); y[:: fs // F0] = 1.0
    for F, B in zip(formants, bws):
        r, th = np.exp(-np.pi * B / fs), 2 * np.pi * F / fs
        y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    return y / np.max(np.abs(y))

VOWELS = [("/i/ (beet)", (270, 2290, 3010)), ("/ɑ/ (father)", (730, 1090, 2440)), ("/u/ (boot)", (300, 870, 2240))]

# 1. spectrograms of three synthetic vowels
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (name, F) in zip(axes, VOWELS):
    f, t, S = spectrogram(synth(F), fs=fs, nperseg=256, noverlap=192)
    ax.pcolormesh(t, f, 10 * np.log10(S + 1e-12), shading="auto", cmap="Blues", vmin=-90)
    for k, Fk in enumerate(F, 1):
        ax.axhline(Fk, color=ORANGE, lw=1.2, ls="--")
        ax.text(t[-1] * 0.98, Fk + 60, f"F{k} = {Fk} Hz", ha="right", color=INK, fontsize=9)
    ax.set_title(name, color=INK); ax.set_xlabel("time (s)"); ax.set_ylim(0, 4000)
axes[0].set_ylabel("frequency (Hz)")
fig.suptitle("Synthetic vowels: the dark horizontal bands are the formants", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_vowel_spectrograms.png"), dpi=130); plt.close(fig)

# 2. vowel space (F2 vs F1, both axes reversed as phoneticians draw it)
PB = {"i": (270, 2290), "ɪ": (390, 1990), "ɛ": (530, 1840), "æ": (660, 1720), "ɑ": (730, 1090),
      "ɔ": (570, 840), "ʊ": (440, 1020), "u": (300, 870), "ʌ": (640, 1190), "ɝ": (490, 1350)}
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for v, (F1, F2) in PB.items():
    ax.scatter(F2, F1, s=70, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"/{v}/", (F2, F1), textcoords="offset points", xytext=(8, 4), color=INK)
ax.invert_xaxis(); ax.invert_yaxis()
ax.set_xlabel("F2 (Hz)  ←  front … back"); ax.set_ylabel("F1 (Hz)  ←  close … open")
ax.set_title("American English vowel space (Peterson & Barney 1952, men)", color=INK, fontsize=11)
ax.grid(color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_vowel_space.png"), dpi=130); plt.close(fig)

# 3. parse explosion: Catalan numbers vs number of PPs
k = np.arange(0, 9)
cat = [math.comb(2 * (m + 1), m + 1) // (m + 2) for m in k]
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(k, cat, color=BLUE, lw=2, marker="o", ms=8, markeredgecolor="white", markeredgewidth=2)
for x, y in zip(k, cat):
    if x in (1, 3, 5, 8): ax.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(-6, 9), color=INK)
ax.set_yscale("log"); ax.set_xlabel("number of prepositional phrases after 'I saw the man'")
ax.set_ylabel("number of parses (log scale)")
ax.set_title("PP-attachment ambiguity grows as the Catalan numbers", color=INK, fontsize=11)
ax.grid(color="#e6e5e1", lw=0.8, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_pp_parse_growth.png"), dpi=130); plt.close(fig)
print("saved", sorted(f for f in os.listdir(OUT) if f.startswith("01_")), "Catalan:", cat)
