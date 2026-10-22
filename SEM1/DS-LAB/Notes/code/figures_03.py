"""Figures for Note 03 (Feature engineering, Titanic). Writes ../images/03_*.png.

Run:  python3 figures_03.py      (needs titanic.csv in this folder)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#ecebe8"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})

raw = pd.read_csv(os.path.join(HERE, "titanic.csv"))
df = raw.copy()
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["FamilyBand"] = pd.cut(df["FamilySize"], bins=[0, 1, 4, 11], labels=["Alone", "Small (2–4)", "Large (5+)"])
t = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip().replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df["Title"] = t.where(t.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
df["HasCabin"] = df["Cabin"].notna().map({True: "Cabin known", False: "No cabin"})
df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 30, 50, 100], labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"])
overall = raw["Survived"].mean()

# 1. Survival rate by raw and engineered features
panels = [("Sex", "Sex (raw)"), ("Pclass", "Pclass (raw)"), ("Title", "Title (from Name)"),
          ("FamilyBand", "Family size (SibSp+Parch+1)"), ("AgeGroup", "AgeGroup (binned Age)"), ("HasCabin", "Cabin known? (from NaN)")]
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for ax, (col, title) in zip(axes.ravel(), panels):
    g = df.groupby(col, observed=True)["Survived"].agg(["mean", "count"])
    if col == "Title":
        g = g.sort_values("mean")
    colors = [BLUE if col in ("Sex", "Pclass") else ORANGE] * len(g)
    bars = ax.bar(g.index.astype(str), g["mean"], color=colors, width=0.65)
    ax.axhline(overall, color=MUTED, lw=1, ls="--")
    for b, (m, n) in zip(bars, g.to_numpy()):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.0%}\nn={int(n)}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
fig.suptitle(f"Survival rate by feature (dashed line = overall {overall:.0%}); orange = engineered",
             x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_survival_by_feature.png"), dpi=150)
plt.close(fig)

# 2. One column, four scalings
fare = raw[["Fare"]]
versions = [("Raw Fare", fare["Fare"].to_numpy()),
            ("StandardScaler  (x − mean) / std", StandardScaler().fit_transform(fare).ravel()),
            ("MinMaxScaler  (x − min) / (max − min)", MinMaxScaler().fit_transform(fare).ravel()),
            ("RobustScaler  (x − median) / IQR", RobustScaler().fit_transform(fare).ravel()),
            ("log1p(Fare)", np.log1p(fare["Fare"]).to_numpy())]
fig, axes = plt.subplots(len(versions), 1, figsize=(8, 8.5))
for ax, (name, x) in zip(axes, versions):
    ax.hist(x, bins=60, color=ORANGE if name.startswith("log") else "#c3c2b7", ec="white")
    ax.axvline(np.median(x), color=BLUE, lw=1.8, ls="--")
    ax.set_title(f"{name}    range {x.min():.2f} … {x.max():.2f}", color=INK, loc="left", fontsize=10)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
fig.suptitle("Linear scalers change the axis, not the shape; a log changes the shape (dashed = median)",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_scalers.png"), dpi=150)
plt.close(fig)

# 3. Age imputation: global median vs median by title
title_raw = raw["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
fills = {"global median (28)": raw["Age"].fillna(raw["Age"].median()),
         "median by Title": raw["Age"].fillna(raw.groupby(title_raw)["Age"].transform("median"))}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
bins = np.arange(0, 82, 2)
for ax, (name, filled) in zip(axes, fills.items()):
    ax.hist(raw["Age"].dropna(), bins=bins, color="#c3c2b7", ec="white", label="known ages (714)")
    ax.hist(filled[raw["Age"].isna()], bins=bins, bottom=np.histogram(raw["Age"].dropna(), bins=bins)[0],
            color=ORANGE, ec="white", label="imputed (177)")
    ax.set_title(name, color=INK, loc="left", fontsize=11)
    ax.set_xlabel("Age")
    ax.legend(frameon=False, fontsize=9)
fig.suptitle("Filling 177 ages with one number builds one tall spike; Title splits it into smaller, more plausible ones", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_age_imputation.png"), dpi=150)
plt.close(fig)
print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("03_")))
