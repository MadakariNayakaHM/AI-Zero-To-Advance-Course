# Applied Mathematics for Data Science & AI

> Dr. Arulalan Rajan · IIIT Dharwad · Semester 1

## Notes

| # | Topic | Lectures | Notebook |
|:-:|:--|:--|:--|
| 01 | [Linear Systems, Determinant & Inverse](01-Linear-Systems-Determinant-Inverse.md) | 16 & 19 Sep | [deep dive](code/01_linear_systems_deep_dive.ipynb) |
| 02 | [Gaussian Elimination, Row Operations & Rank](02-Gaussian-Elimination-Row-Operations-Rank.md) | 19 & 23 Sep | [deep dive](code/02_elimination_deep_dive.ipynb) |
| 03 | [Vector Spaces & Subspaces](03-Vector-Spaces-and-Subspaces.md) | 26 & 30 Sep | [deep dive](code/03_vector_spaces_deep_dive.ipynb) |
| 04 | [Span, Linear Independence, Basis & Dimension](04-Span-Independence-Basis-Dimension.md) | 30 Sep & 3 Oct | [deep dive](code/04_basis_deep_dive.ipynb) |
| 05 | [Null Space & Nullity](05-Null-Space-and-Nullity.md) | 3 Oct | [deep dive](code/05_null_space_deep_dive.ipynb) |
| 06 | [Problem-Solving Session: Complete Solution, Column Space & Rank](06-Problem-Solving-Column-Space-and-Rank.md) | 5 & 7 Oct | [deep dive](code/06_column_space_deep_dive.ipynb) |
| 07 | [Linear Transformations](07-Linear-Transformations.md) | 7 & 10 Oct | [deep dive](code/07_linear_transformations_deep_dive.ipynb) |
| 08 | [Dot Product, Projections & Least Squares](08-Dot-Product-Projections-Least-Squares.md) | 10 Oct | [deep dive](code/08_projections_deep_dive.ipynb) |

Each note has full proofs, worked examples, a bank of 25+ practice problems with full solutions, and real-world case studies.

- **Class examples notebook:** [linear algebra, part 1](code/linear_algebra_part1.ipynb) verifies every example from the lectures and includes a "find the redundant features in a dataset" demo.
- **Source:** [the handwritten lecture notes, typed](00-Handwritten-Notes-Transcribed.md) (pages 1–81, searchable).

**Reading assignment (7 Oct):** linear combinations, independence, vector spaces, basis · span · the four fundamental subspaces of an $`m\times n`$ matrix (Notes 04–06).

**Coming up:** eigenvalues & eigenvectors · rest of linear algebra · probability & statistics · calculus & optimisation

## Clarifications to the handwritten notes

| Page | Written | Correct reading |
|:-:|:--|:--|
| 13 | Gaussian elimination is "iterative" | It is a direct, finite-step method |
| 28 | Rank = largest square *matrix* with non-zero determinant | Largest square **sub-matrix** |
| 33 | Vector space defined by three conditions | Those three conditions are the *subspace test*; the full definition has ten axioms |
| 36 | Line `y = kx` with `k` free | `k` must be **fixed**, otherwise the set is not closed under addition |
| 40 | All of Ex 3–8 are subspaces | Ex 7 (`y = 3`) is not |
| 49 | Basis of `{0}` written as `{{ }}` | The basis is the empty set `{ }` |
| 58 | $`x_4 = -x_2`$ | $`x_1 = -x_2`$ |
| 61 | Pivots counted "in RREF", but the REF is shown | Same count; the RREF is $`\begin{bmatrix}1&1&0\\0&0&1\end{bmatrix}`$ |
| 71 | $`T(\mathbf 0)=\mathbf 0`$ listed as a separate condition | It follows from homogeneity with $`c = 0`$ |
| 73 | Lines → lines, planes → planes | Only for one-to-one maps; a projection can collapse a line to $`\mathbf 0`$ |
| 76 | A projection must be symmetric and idempotent | $`P^2 = P`$ defines a projection; symmetric as well makes it an **orthogonal** projection |
| 81 | $`(X^\top X)^{-1}`$ labelled the pseudo-inverse | The pseudo-inverse is $`(X^\top X)^{-1}X^\top`$ (full column rank) |

Each one is explained in the relevant note.

## References

Strang, *Introduction to Linear Algebra* · Boyd & Vandenberghe, *Introduction to Applied Linear Algebra* · Farin & Hansford, *Practical Linear Algebra* · NPTEL [*Linear Algebra Through Geometry*](https://nptel.ac.in/courses/106108482)
