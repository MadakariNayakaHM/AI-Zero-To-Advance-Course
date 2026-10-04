# Applied Mathematics for Data Science & AI: Notes

**Instructor:** Dr. Arulalan Rajan · M.Tech Sem 1 (Sep 2026 – )
**Plan (from class):** Linear Algebra (wrapping up ~10 Oct) → Probability & Statistics / Calculus & Optimisation
**Reference books (notes p.19):** Farin & Hansford *Practical Linear Algebra* · Gilbert Strang *Linear Algebra* (5th ed.) · Stephen Boyd *Intro to Applied Linear Algebra* · NPTEL *Advanced Matrix Theory* (Prof. Vittal Rao) · NPTEL [*Linear Algebra Through Geometry*](https://nptel.ac.in/courses/106108482) (Ashok Rao & Arulalan Rajan)

## How each note is organised
Big picture → intuition & analogies (incl. the professor's) → definitions & theory → worked examples (all from class, verified in code) → ML connections → 🎓 professor emphasised → ⚠️ common confusions → 📝 practice problems with answers → 🧾 cheat sheet → 📚 curated links.

Levels: 🟢 basic · 🟡 intermediate · 🔴 advanced / beyond syllabus.

## Lecture tracker

| # | Note | Date(s) | Sources | Status |
|---|---|---|---|---|
| 00 | [Handwritten notes: typed transcription](00-Handwritten-Notes-Transcribed.md) | 16 Sep – 3 Oct | PDF pp. 1–55 | ✅ |
| 01 | [Linear Systems, Determinant & Inverse](01-Linear-Systems-Determinant-Inverse.md) | 16 Sep (L1), 19 Sep (L2) | Notes pp. 1–12 | ✅ (no transcript) |
| 02 | [Gaussian Elimination, Row Operations & Rank](02-Gaussian-Elimination-Row-Operations-Rank.md) | 19 Sep (L2), 23 Sep (L3) | Notes pp. 13–29 | ✅ (no transcript) |
| 03 | [Vector Spaces & Subspaces](03-Vector-Spaces-and-Subspaces.md) | 26 Sep, 30 Sep | Notes pp. 30–41 + 30 Sep transcript | ✅ |
| 04 | [Span, Linear Independence, Basis & Dimension](04-Span-Independence-Basis-Dimension.md) | 30 Sep, 3 Oct | Notes pp. 42–50 + transcripts | ✅ |
| 05 | [Null Space & Nullity](05-Null-Space-and-Nullity.md) | 3 Oct | Notes pp. 50–55 + 3 Oct transcript | ✅ |
| 06 | Problem-solving session | **Mon 5 Oct, 8–9:30 pm** | — | ⏳ |
| 07 | (Column space, rank–nullity, eigen…?) | Wed 7 Oct, Sat 10 Oct | — | ⏳ |

**Notebook:** [`code/linear_algebra_part1.ipynb`](code/linear_algebra_part1.ipynb) verifies every handwritten example (NumPy / SymPy), plus a "find redundant features in a dataset" demo.
**Figures:** [`code/make_figures.py`](code/make_figures.py) → `images/`

## ⚠️ Slips found in the handwritten notes
None of these change the main ideas; each is explained where it occurs.

| Page | Written | Correct reading | Note |
|---|---|---|---|
| 5 | Brace labels $\frac{1}{a_{11}a_{22}-a_{21}a_{12}}$ as $\det(A)$ | Only the denominator is $\det(A)$ | [01 §6](01-Linear-Systems-Determinant-Inverse.md#6-the-inverse-matrix-) |
| 13 | Gaussian elimination: "Iterative" | It's a **direct** (finite-step) method | [02 §1](02-Gaussian-Elimination-Row-Operations-Rank.md#1-big-picture) |
| 20 | "=" between the original and row-swapped matrices | Row-**equivalent** ($\sim$) | [02 §6](02-Gaussian-Elimination-Row-Operations-Rank.md#6-elementary-row-operations--matrices-) |
| 28 | "largest square matrix with det ≠ 0" | Largest square **sub-matrix** | [02 §9](02-Gaussian-Elimination-Row-Operations-Rank.md#9-rank-of-a-matrix-) |
| 29 | "det = 0 ⇒ Rank = 1" | det = 0 ⇒ rank < 2; it's 1 because of a non-zero entry | [02 §9](02-Gaussian-Elimination-Row-Operations-Rank.md#9-rank-of-a-matrix-) |
| 29 | Galois "21" | He died aged 20 (1811–1832) | [02 §10](02-Gaussian-Elimination-Row-Operations-Rank.md#10-side-note-galois--finite-fields-) |
| 33/39 | Vector space = 3 conditions | That's the **subspace test**; the full definition has ~10 axioms | [03 §4](03-Vector-Spaces-and-Subspaces.md#4-definition-of-a-vector-space-) |
| 35 | Second vector labelled $\vec u$ | Should be $\vec v$ | [03 §5](03-Vector-Spaces-and-Subspaces.md#5-examples--non-examples-) |
| 36 | $S_4$: "$x_1\in\mathbb R,\ k\in\mathbb R$" | $k$ must be **fixed** (otherwise not closed) | [03 §5](03-Vector-Spaces-and-Subspaces.md#5-examples--non-examples-) |
| 40 | "All sets Ex 3–8 are subspaces" | Ex 7 ($y=3$) is **not** | [03 §6](03-Vector-Spaces-and-Subspaces.md#6-vector-subspaces-) |
| 47 | "n l.i. vectors form a basis of an n-dim space" | …if they lie **in** that space | [04 §8](04-Span-Independence-Basis-Dimension.md#8-the-professors-comments-remarks-15-) |
| 49 | Basis of $\{\mathbf 0\}$ = $\{\{\ \}\}$ | $\{\ \} = \varnothing$ | [04 §9](04-Span-Independence-Basis-Dimension.md#9-puzzle-basis-of-the-zero-space-) |

## Professor's teaching notes

- **Participate actively**: the professor explicitly wants everyone (not just the same 4–5 voices) to answer.
- Doubts → post in the WhatsApp group.
- Extra material → his NPTEL course *Linear Algebra Through Geometry* (also on YouTube).

## Running the code
```bash
pip install numpy scipy sympy matplotlib jupyter
cd code
python make_figures.py
jupyter notebook linear_algebra_part1.ipynb
```
