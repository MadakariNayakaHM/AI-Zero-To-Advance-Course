# 03 · Vectors, Vector Spaces & Subspaces

> **Course:** Applied Mathematics for Data Science & AI · Dr. Arulalan Rajan
>
> **Lectures:** 26 Sep 2026 (vectors, vector spaces) and 30 Sep 2026, first part (subspaces & their geometry)
>
> **Sources:** handwritten notes pp. 30–41 and the 30 Sep class transcript
>
> **Notebook:** [`code/linear_algebra_part1.ipynb`](code/linear_algebra_part1.ipynb), Part D

---

## 📌 Table of Contents

1. [Big Picture](#1-big-picture)
2. [What is a Vector?](#2-what-is-a-vector-)
3. [The Two Operations](#3-the-two-operations-)
4. [Definition of a Vector Space](#4-definition-of-a-vector-space-)
5. [Examples & Non-Examples](#5-examples--non-examples-)
6. [Vector Subspaces](#6-vector-subspaces-)
7. [Geometry of Subspaces](#7-geometry-of-subspaces-)
8. [Vector Spaces in ML](#8-vector-spaces-in-ml-)
9. [Professor Emphasised](#9--professor-emphasised)
10. [Common Confusions](#10--common-confusions)
11. [Practice Problems](#11--practice-problems)
12. [Cheat Sheet](#12--cheat-sheet)
13. [Go Deeper](#13--go-deeper-curated-links)

---

## 1. Big Picture

Notes 01–02 treated a matrix as a **table** and solved equations. Now we zoom in on the **objects** the matrix acts on, *vectors*, and on the **"universes" they live in**, *vector spaces*.

Why bother with the abstraction? In ML, **everything becomes a vector**: a house (features), a word (embedding), an image (pixels), a user (preferences). Once data lives in a vector space, we can **add**, **scale**, **measure distance and similarity**, and **project**. That is all of ML geometry.

---

## 2. What is a Vector? 🟢

| View | Description | Example |
|---|---|---|
| **Physics / geometry** | An arrow with **direction and magnitude** | Velocity of a car |
| **Algebra** | An **ordered list of numbers** (ordered pair, n-tuple) | $`\begin{pmatrix}2\\3\end{pmatrix}`$, a 2-component vector |
| **Data science** | A **row of the data matrix**: one observation | (area, rooms, age) of a house |
| **Abstract (this course)** | **Any element of a vector space** (§4) | Even matrices and polynomials! |

- 2-D vector: $`\begin{pmatrix}x\\y\end{pmatrix}`$ with $`x, y \in \mathbb{R}`$. It's drawn as an arrow from the origin to the point $`(x, y)`$.
- **Ordered** matters: $`(2,3) \ne (3,2)`$.
- An ordered $`n`$-tuple $`(x_1, \dots, x_n)`$ is a vector in $`\mathbb{R}^n`$.

**Zero vector** $`\mathbf{0} = (0, 0)`$: all components zero. It has **no direction** (asked in class: *"why call it a vector if it has no magnitude or direction?"* Professor: it's simply the vector whose components are all 0, and it gets a special name because it plays a special role).

---

## 3. The Two Operations 🟢

Think of a "basket" $`\mathcal{V}`$ of elements.

### Vector addition (component-wise)
```math
\vec{u} + \vec{v} = \begin{pmatrix}u_1\\u_2\end{pmatrix} + \begin{pmatrix}v_1\\v_2\end{pmatrix} = \begin{pmatrix}u_1+v_1\\u_2+v_2\end{pmatrix}
```
Example: $`\begin{pmatrix}4\\2\end{pmatrix} + \begin{pmatrix}2\\3\end{pmatrix} = \begin{pmatrix}6\\5\end{pmatrix}`$. Geometrically this is the **parallelogram rule**: place the arrows tip-to-tail, or take the diagonal of the parallelogram they span.

If $`\vec{u} + \vec{v} \in \mathcal{V}`$ for all $`\vec{u}, \vec{v} \in \mathcal{V}`$, then $`\mathcal{V}`$ is **closed under vector addition**.

### Scalar multiplication
```math
\alpha\vec{u} = \begin{pmatrix}\alpha u_1\\ \alpha u_2\end{pmatrix},\qquad 5\begin{pmatrix}2\\3\end{pmatrix} = \begin{pmatrix}10\\15\end{pmatrix},\quad -3\begin{pmatrix}2\\3\end{pmatrix} = \begin{pmatrix}-6\\-9\end{pmatrix}
```
Geometrically: **stretch/shrink** the arrow; a negative scalar **flips** it.

If $`\alpha\vec{u} \in \mathcal{V}`$ for every real $`\alpha`$ and every $`\vec{u} \in \mathcal{V}`$, then $`\mathcal{V}`$ is **closed under scalar multiplication**.

> **Closure** = "you can't escape the basket by adding or scaling".

---

## 4. Definition of a Vector Space 🟢→🟡

**Course definition (notes p.33, p.39):** $`\mathcal{V}(+,\cdot)`$ over the real numbers is a **vector space** if

- **(a)** it is closed under vector addition,
- **(b)** it is closed under scalar multiplication, and
- **(c)** it contains the **zero vector**.

Every element of a vector space is called a **vector**.

### 🔴 The full (textbook) definition
The 3 conditions above are exactly the **subspace test**. They suffice when your set sits inside a known vector space like $`\mathbb{R}^n`$ (which is all this course needs). The complete abstract definition requires, for all $`\vec{u}, \vec{v}, \vec{w} \in \mathcal{V}`$ and scalars $`a, b`$:

| # | Axiom | |
|---|---|---|
| 1 | $`\vec{u} + \vec{v} \in \mathcal{V}`$ | closure (+) |
| 2 | $`\vec{u} + \vec{v} = \vec{v} + \vec{u}`$ | commutativity |
| 3 | $`(\vec{u} + \vec{v}) + \vec{w} = \vec{u} + (\vec{v} + \vec{w})`$ | associativity |
| 4 | $`\exists\, \mathbf{0}: \vec{u} + \mathbf{0} = \vec{u}`$ | zero vector (additive identity) |
| 5 | $`\exists\, {-\vec{u}}: \vec{u} + (-\vec{u}) = \mathbf{0}`$ | additive inverse |
| 6 | $`a\vec{u} \in \mathcal{V}`$ | closure (·) |
| 7 | $`a(\vec{u} + \vec{v}) = a\vec{u} + a\vec{v}`$ | distributivity |
| 8 | $`(a + b)\vec{u} = a\vec{u} + b\vec{u}`$ | distributivity |
| 9 | $`a(b\vec{u}) = (ab)\vec{u}`$ | compatibility |
| 10 | $`1\vec{u} = \vec{u}`$ | identity scalar |

For subsets of $`\mathbb{R}^n`$ with the usual operations, axioms 2, 3, 7–10 are inherited automatically, and closure under scalar multiplication gives 4–5 (take $`a = 0`$ and $`a = -1`$). That's why the 3-condition test works.

> 💡 **Quick zero-vector check:** if $`\mathbf{0} \notin \mathcal{V}`$, stop: it's not a vector space. This is the fastest way to rule things out.

---

## 5. Examples & Non-Examples 🟢

| # | Set | Vector space? | Why |
|---|---|---|---|
| Ex 1 | $`\mathbb{R}`$ (real line) | ✅ | Sum and multiples of reals are real; contains 0 |
| Ex 2 | $`\mathbb{R}^2 = \mathbb{R}\times\mathbb{R}`$ (xy-plane) | ✅ | $`3(2,3) = (6,9)`$, $`(4,2)+(2,3) = (6,5)`$, … |
| Ex 3 | $`S_1 = \{(x_1, x_1) : x_1\in\mathbb{R}\}`$ (line $`y = x`$) | ✅ | Proof below |
| Ex 4 | $`S_2 = \{(x_1, 0)\}`$ (x-axis) | ✅ | Line through the origin |
| Ex 5 | $`S_3 = \{(0, x_2)\}`$ (y-axis) | ✅ | Line through the origin |
| Ex 6 | $`S_4 = \{(x_1, kx_1)\}`$ for a **fixed** $`k`$ (line $`y = kx`$) | ✅ | Line through the origin |
| Ex 7 | $`S_5 = \{(x_1, 3)\}`$ (line $`y = 3`$) | ❌ | $`\mathbf{0} \notin S_5`$; also $`(1,3)+(2,3) = (3,6)\notin S_5`$ |
| Ex 8 | $`S_6 = \{(0,0)\}`$ (just the origin) | ✅ | $`0+0=0`$, $`k\cdot0 = 0`$: the **zero space** |
| Ex 9 | $`\mathcal{M}^{2\times2}`$, all 2×2 real matrices | ✅ | Proof below. "Vectors" needn't be arrows! |

### Proof for Ex 3 (template for any such proof)
Let $`\vec{u} = (u_1, u_1)`$, $`\vec{v} = (v_1, v_1) \in S_1`$, and $`k\in\mathbb{R}`$.

1. **Addition:** $`\vec{u} + \vec{v} = (u_1+v_1,\ u_1+v_1)`$. Both components are equal, so it is in $`S_1`$ ✓
2. **Scalar:** $`k\vec{u} = (ku_1, ku_1)`$. Both components are equal, so it is in $`S_1`$ ✓
3. **Zero:** $`(0,0)`$ has equal components, so it is in $`S_1`$ ✓

Hence $`S_1`$ is a vector space. ∎

> 📝 Small fixes to the handwritten notes. (p.35) The second vector is labelled $`\vec{u}`$ but should be $`\vec{v} = (v_1, v_1)`$. (p.36) Ex 6 is written with "$`x_1\in\mathbb{R},\ k\in\mathbb{R}`$". It must be read as **one fixed $`k`$**. If $`k`$ also varies, the set becomes the union of all lines through the origin, which is **not** closed under addition: $`(1,1) + (-1,1) = (0,2)`$ lies on neither line. (Notebook Part D shows this.)

### Proof for Ex 9: matrices are vectors too
$`A + B = \begin{bmatrix}a_1+b_1&a_2+b_2\\a_3+b_3&a_4+b_4\end{bmatrix} \in \mathcal{M}^{2\times2}`$, $`kA \in \mathcal{M}^{2\times2}`$, and the zero matrix $`\in \mathcal{M}^{2\times2}`$. ✓

**Big idea:** "vector" means **anything you can add and scale** following the rules. Other vector spaces: polynomials of degree ≤ n; all functions $`f:\mathbb{R}\to\mathbb{R}`$; audio signals; images (a 28×28 image is a vector in $`\mathbb{R}^{784}`$).

---

## 6. Vector Subspaces 🟢

**Q (30 Sep):** *Does $`S_1`$ (line $`y=x`$) contain all points of $`\mathbb{R}^2`$?* **No.** But it is a **subset** of $`\mathbb{R}^2`$ that is closed under +, ·, and contains $`\mathbf{0}`$.

> **Definition:** Any **subset** of a vector space which **by itself is a vector space** is called a **vector subspace**.

So Ex 3, 4, 5, 6 and 8 are subspaces of $`\mathbb{R}^2`$. (The notes say "all sets from Ex 3 to Ex 8", but **Ex 7 is excluded**: it's not a vector space.)

### Is ℝ² a subspace of itself? (class discussion)
**Yes.** Set-theoretic reason (professor): **every set is a subset of itself**. Since $`\mathbb{R}^2`$ is a vector space and $`\mathbb{R}^2 \subseteq \mathbb{R}^2`$, it is a subspace of itself. "Sub" does **not** mean "strictly smaller".

> ⚠️ A student suggested "put all subspaces together → you get $`\mathbb{R}^2`$". The professor rejected this reasoning: *"you can't just put everything from the supermarket in a bag and call it a subspace."* **A union of subspaces is generally NOT a subspace** (see the Ex 6 fix above). The **intersection** of subspaces always is.

---

## 7. Geometry of Subspaces 🟢→🟡

![subspaces of R2](images/06_subspaces_R2.png)

**All subspaces of $`\mathbb{R}^2`$:**

| Subspace | Geometry | Dimension |
|---|---|---|
| $`\{\mathbf{0}\}`$ | The origin | 0 |
| $`\{t\vec{v}\}`$ | Any **line through the origin** | 1 |
| $`\mathbb{R}^2`$ | The whole plane | 2 |

That's the complete list. There are no others.

**Generalisation to $`\mathbb{R}^n`$ (professor):**

- Any line through the origin in $`\mathbb{R}^n`$ is a subspace.
- Any plane through the origin is a subspace of a larger vector space (e.g. of $`\mathbb{R}^3`$).
- In general, subspaces of $`\mathbb{R}^n`$ are "flat" objects **through the origin** of dimension 0, 1, …, n.

**Why must subspaces pass through the origin?** Closure under scalar multiplication with $`\alpha = 0`$ forces $`0\cdot\vec{u} = \mathbf{0}`$ to be inside. A line **not** through the origin (like $`y = 3`$) has no zero vector (*"additive identity is not there"*), so it is never a subspace. (It's an *affine* set: a shifted subspace.)

**A single non-zero vector is never a vector space** (professor): $`\{(1,1)\}`$ fails, since $`2(1,1) = (2,2)\notin`$ the set. Only $`\{\mathbf{0}\}`$ works as a one-element vector space.

**Infinitely many points:** every subspace except $`\{\mathbf{0}\}`$ contains **infinitely many** vectors. That sets up the next question: *how do we describe an infinite set with finitely many vectors?* → **span and basis** ([Note 04](04-Span-Independence-Basis-Dimension.md)).

---

## 8. Vector Spaces in ML 🟡

The professor tied it together on 30 Sep: **"Vector space corresponds to feature space."**

| Linear-algebra concept | ML meaning |
|---|---|
| Vector in $`\mathbb{R}^n`$ | One data point with $`n`$ features |
| Vector space $`\mathbb{R}^n`$ | **Feature space** |
| Subspace | The low-dimensional region where data actually lives (e.g. faces occupy a tiny "subspace" of all possible images) |
| Addition / scaling | Combining embeddings: `king − man + woman ≈ queen` |
| Zero vector / origin | Reference point; after mean-centering, the data's mean |
| $`\mathcal{M}^{m\times n}`$ as a vector space | Neural-network weight matrices: gradient descent *adds* scaled gradients, $`W \leftarrow W - \alpha\nabla W`$, which needs closure! |
| Function spaces | Kernels / SVMs work in (infinite-dimensional) function spaces |

> 🔴 **PCA preview:** PCA finds the best **low-dimensional subspace** (through the mean) to project data onto. "Subspace" is the core object of dimensionality reduction.

---

## 9. 🎓 Professor Emphasised

1. Vector space = closed under **addition** and **scalar multiplication**, and **contains the zero vector**.
2. **Subspace** = a subset that is itself a vector space.
3. Subspaces of $`\mathbb{R}^2`$: **origin, lines through origin, $`\mathbb{R}^2`$**. Generalise: planes through origin, etc.
4. **Lines not through the origin are never vector spaces** (no zero vector).
5. Every set is a subset of itself, so $`\mathbb{R}^n`$ is a subspace of $`\mathbb{R}^n`$.
6. A single non-zero vector can't form a vector space.
7. Geometry helps in higher dimensions; build the picture in 2-D first.
8. **Vector space ↔ feature space** in ML.
9. Class participation: the professor strongly wants **everyone** to answer, not just a few. Unmute and respond!

---

## 10. ⚠️ Common Confusions

| Confusion | Clarification |
|---|---|
| A subspace must be smaller | $`V`$ is a subspace of itself; $`\{\mathbf{0}\}`$ is the smallest |
| Any line is a subspace | Only lines **through the origin** |
| Union of subspaces is a subspace | Generally **no** (intersection yes) |
| "Vector" = arrow with numbers | Any element of a vector space (matrices, functions…) |
| $`\{(1,1)\}`$ is a vector space | Not closed under scaling; only $`\{\mathbf{0}\}`$ works as a single point |
| 3 conditions = full definition | They're the subspace test; the full definition has ~10 axioms |
| Zero vector = "no vector" | It's a real vector, $`(0,\dots,0)`$; $`\{\mathbf{0}\}`$ is a genuine (0-D) space |

---

## 11. 📝 Practice Problems

<details>
<summary><b>P1.</b> Is $`W = \{(x, y) : y = 2x + 1\}`$ a subspace of $`\mathbb{R}^2`$?</summary>

No: $`(0,0)`$ fails $`0 = 2·0 + 1`$. (It's a line not through the origin.)
</details>

<details>
<summary><b>P2.</b> Is $`W = \{(x, y, z) : x + y + z = 0\}`$ a subspace of $`\mathbb{R}^3`$? What is it geometrically?</summary>

Yes. If $`x_1+y_1+z_1 = 0`$ and $`x_2+y_2+z_2 = 0`$, the sum satisfies it; $`k(x+y+z) = 0`$; and $`(0,0,0)`$ works. Geometrically it is a **plane through the origin** with normal $`(1,1,1)`$.
</details>

<details>
<summary><b>P3.</b> Is the first quadrant $`\{(x,y): x\ge0, y\ge0\}`$ a subspace?</summary>

No: closed under addition and contains 0, but $`(-1)(1,1) = (-1,-1)`$ escapes. Fails scalar closure.
</details>

<details>
<summary><b>P4.</b> Is $`\{(x,y): xy = 0\}`$ (the union of both axes) a subspace?</summary>

No: $`(1,0) + (0,1) = (1,1)`$, and $`1·1 \ne 0`$. A union of two subspaces is not a subspace.
</details>

<details>
<summary><b>P5.</b> Is the set of 2×2 matrices with determinant 0 a subspace of $`\mathcal{M}^{2\times2}`$?</summary>

No: $`\begin{bmatrix}1&0\\0&0\end{bmatrix} + \begin{bmatrix}0&0\\0&1\end{bmatrix} = I`$, with $`\det = 1`$. Not closed under addition.
</details>

<details>
<summary><b>P6.</b> Is the set of 2×2 <i>symmetric</i> matrices ($`A = A^\top`$) a subspace?</summary>

Yes: $`(A+B)^\top = A^\top + B^\top = A + B`$, $`(kA)^\top = kA`$, and $`0^\top = 0`$.
</details>

<details>
<summary><b>P7.</b> Is the solution set of $`Ax = b`$ with $`b \ne 0`$ a subspace? What about $`Ax = 0`$?</summary>

$`Ax=b`$ ($`b\neq0`$): no, since $`x = 0`$ gives $`A0 = 0 \ne b`$. $`Ax = 0`$: **yes**, this is the null space ([Note 05](05-Null-Space-and-Nullity.md)).
</details>

---

## 12. 🧾 Cheat Sheet

- **Vector:** ordered n-tuple / arrow / any element of a vector space.
- **Vector space test (subsets of $`\mathbb{R}^n`$):** ① $`\mathbf{0} \in V`$ ② $`u+v\in V`$ ③ $`ku\in V`$.
- Check the zero vector first: if it's missing, the set is not a vector space.
- **Subspace:** a subset that's itself a vector space. Subspaces of $`\mathbb{R}^2`$ are $`\{\mathbf{0}\}`$, lines through 0, and $`\mathbb{R}^2`$.
- Lines/planes **not** through the origin are not subspaces. Unions usually aren't either; intersections are.
- Matrices, polynomials and functions are vectors too.
- ML: vector space = feature space.

---

## 13. 📚 Go Deeper: Curated Links

| Topic | Why | Link |
|---|---|---|
| What is a vector? | 3 views (physics, CS, maths) unified | [3Blue1Brown — Vectors, Ch. 1](https://www.youtube.com/watch?v=fNk_zzaMoSs) |
| Abstract vector spaces | Functions as vectors | [3Blue1Brown — Abstract vector spaces, Ch. 16](https://www.youtube.com/watch?v=TgKwz5Ikpc8) |
| Subspaces & column space | Strang's treatment | [MIT 18.06 — L6: Column Space and Nullspace](https://www.youtube.com/watch?v=8o5Cmfpeo6g) |
| The professor's NPTEL course | Vector spaces via geometry | [NPTEL — Linear Algebra Through Geometry](https://nptel.ac.in/courses/106108482) |
| Subspaces, interactively | Free online textbook with demos | [Interactive Linear Algebra, §2.6 Subspaces](https://textbooks.math.gatech.edu/ila/subspaces.html) |
| Visual tool used in class | Plot vectors/lines yourself | [GeoGebra Calculator](https://www.geogebra.org/calculator) |
| Embeddings as vectors | Why "vector space = feature space" matters in NLP | [Jay Alammar — The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) |
| Rigorous treatment | Ch. 2.4 Vector Spaces | [Mathematics for Machine Learning (free)](https://mml-book.github.io/) |

---
⬅️ [02 · Gaussian Elimination & Rank](02-Gaussian-Elimination-Row-Operations-Rank.md) · [Index](README.md) · ➡️ [04 · Span, Independence, Basis, Dimension](04-Span-Independence-Basis-Dimension.md)
