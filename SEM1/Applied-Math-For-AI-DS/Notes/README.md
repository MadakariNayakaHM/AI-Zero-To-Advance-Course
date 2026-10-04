# Applied Mathematics for Data Science & AI

> Dr. Arulalan Rajan · IIIT Dharwad · Semester 1

## Notes

| # | Topic | Lectures |
|:-:|:--|:--|
| 01 | [Linear Systems, Determinant & Inverse](01-Linear-Systems-Determinant-Inverse.md) | 16 & 19 Sep |
| 02 | [Gaussian Elimination, Row Operations & Rank](02-Gaussian-Elimination-Row-Operations-Rank.md) | 19 & 23 Sep |
| 03 | [Vector Spaces & Subspaces](03-Vector-Spaces-and-Subspaces.md) | 26 & 30 Sep |
| 04 | [Span, Linear Independence, Basis & Dimension](04-Span-Independence-Basis-Dimension.md) | 30 Sep & 3 Oct |
| 05 | [Null Space & Nullity](05-Null-Space-and-Nullity.md) | 3 Oct |

- **Notebook:** [linear algebra, part 1](code/linear_algebra_part1.ipynb) verifies every example from the lectures and includes a "find the redundant features in a dataset" demo.
- **Source:** [the handwritten lecture notes, typed](00-Handwritten-Notes-Transcribed.md) (pages 1–55, searchable).

**Coming up:** problem-solving session · rest of linear algebra · probability & statistics · calculus & optimisation

## Clarifications to the handwritten notes

| Page | Written | Correct reading |
|:-:|:--|:--|
| 13 | Gaussian elimination is "iterative" | It is a direct, finite-step method |
| 28 | Rank = largest square *matrix* with non-zero determinant | Largest square **sub-matrix** |
| 33 | Vector space defined by three conditions | Those three conditions are the *subspace test*; the full definition has ten axioms |
| 36 | Line `y = kx` with `k` free | `k` must be **fixed**, otherwise the set is not closed under addition |
| 40 | All of Ex 3–8 are subspaces | Ex 7 (`y = 3`) is not |
| 49 | Basis of `{0}` written as `{{ }}` | The basis is the empty set `{ }` |

Each one is explained in the relevant note.

## References

Strang, *Introduction to Linear Algebra* · Boyd & Vandenberghe, *Introduction to Applied Linear Algebra* · Farin & Hansford, *Practical Linear Algebra* · NPTEL [*Linear Algebra Through Geometry*](https://nptel.ac.in/courses/106108482)
