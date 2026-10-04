# Handwritten Lecture Notes: Typed Transcription (pp. 1–55)

> Faithful transcription of `Class-PPT/Lecture notes till 3rd October.pdf` (16 Sep – 3 Oct 2026) into Markdown + LaTeX, so it can be searched and copied.
>
> Content is kept as written. `[CHECK: …]` marks a suspected slip in the original; each one is discussed in the topic notes (01–05).
>
> The study notes are in [README.md](README.md). This file is the raw source.


## Page 1

**Applied Mathematics for DS & AI**

**LECTURE - 1.**
**16.09.2026.**

- **Matrix:** Table of rows & cols. $`m`$ rows, $`n`$ cols.
  - Every row: One observation of $`n`$-variables
  - Every col: $`m`$ observations of a single variable

**Ex. 1.**

- Solve

```math
\begin{aligned}
2x + y &= 3 \\
x + 2y &= 3
\end{aligned}
\quad\Rightarrow\quad
\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
=
\begin{bmatrix} 3 \\ 3 \end{bmatrix}
```
(written under the matrices: $`A \quad x \quad = \quad b`$.)

$`x = 1,\ y = 1`$ solves the above.

## Page 2

**Ex 2**

- Solve:

```math
\begin{aligned}
2x + y &= 3 \\
4x + 2y &= 6
\end{aligned}
\;\Bigg]\rightarrow\; y = 3 - 2x
```
(Annotation with arrow: "Both eqns are the same".)

Soln is given by:

| $`x`$ | 1 | 0 | 1.5 | 2 | ... |
|---|---|---|---|---|---|
| $`y`$ | 1 | 3 | 0 | −1 | ... |

a representative structure:

```math
\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x \\ 3 - 2x \end{pmatrix}
```

Q: Why did we get infinitely many soln? in this case?
→ Because the eqns are redundant.

- **Ex: 3** Solve

```math
\begin{aligned}
2x + y &= 3 \\
2x + y &= 4
\end{aligned}
```

Q: How did we land in this situation? $`\Downarrow`$ Measurement inconsistency

<u>No</u> <u>soln</u> $`\Rightarrow`$ No soln can never be a soln.

$`\Rightarrow`$ We look for least <u>squared</u> soln. $`\rightarrow`$ Variance being minimized.

$`\Rightarrow`$ Approximate <u>Soln</u>.

$`\Rightarrow`$ The soln that takes you to a destination closer to the actual one.

## Page 3

**Q: Given a system $`Ax = b`$, how do we find the soln?**

```math
\begin{aligned}
a_{11}x_1 + a_{12}x_2 &= b_1 \quad \rightarrow (1) \\
a_{21}x_1 + a_{22}x_2 &= b_2 \quad \rightarrow (2)
\end{aligned}
\quad\Rightarrow\quad
\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
```
(written under: $`A \quad x \quad = \quad b`$)

Eliminate $`x_2`$ from both eqns.

Multiply (1) by $`a_{22}`$ & (2) by $`a_{12}`$ & Subtract

```math
\begin{aligned}
\Rightarrow\ a_{11}a_{22}x_1 + a_{12}a_{22}x_2 &= a_{22}b_1 \\
\ominus\ \ a_{12}a_{21}x_1 \ \ominus\ a_{12}a_{22}x_2 &= \ominus\ a_{12}b_2
\end{aligned}
```
[The $`x_2`$ terms are struck through/cancelled with a line; signs of the second row circled as "−" for subtraction.]

```math
x_1(a_{11}a_{22} - a_{12}a_{21}) = b_1 a_{22} - b_2 a_{12}.
```

```math
\rightarrow\quad x_1 = \frac{b_1 a_{22} - b_2 a_{12}}{a_{11}a_{22} - a_{12}a_{21}} \qquad \text{provided } a_{11}a_{22} - a_{12}a_{21} \neq 0.
```

## Page 4

```math
x_2 = \frac{b_2 a_{11} - b_1 a_{21}}{a_{11}a_{22} - a_{12}a_{21}} \qquad \text{provided the denominator} \neq 0
```

**Determinant of $`A`$** $`= a_{11}a_{22} - a_{12}a_{21}`$

```math
\Rightarrow A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
```

[Diagram: input vector $`(x_1, x_2)^\top`$ → box labelled $`A`$ → output vector $`(b_1, b_2)^\top`$.]

## Page 5

```math
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix}
(b_1 a_{22} - b_2 a_{12}) / (a_{11}a_{22} - a_{21}a_{12}) \\
(a_{11} b_2 - a_{21} b_1) / (a_{11}a_{22} - a_{21}a_{12})
\end{bmatrix}
```
(The subscript in "$`b_2 a_{12}`$" in the first row is cramped and looks like "$`a_2`$"; tick marks above the denominator.)

```math
\Rightarrow \underbrace{\frac{1}{(a_{11}a_{22} - a_{21}a_{12})}}_{\det(A)}
\begin{bmatrix} b_1 a_{22} - b_2 a_{12} \\ -a_{21} b_1 + a_{11} b_2 \end{bmatrix}
```
[CHECK: The underbrace labels the whole fraction $`\frac{1}{a_{11}a_{22}-a_{21}a_{12}}`$ as $`\det(A)`$; strictly only the denominator $`a_{11}a_{22}-a_{21}a_{12}`$ is $`\det(A)`$ (the brace is drawn under the denominator, so this is probably what was meant).]

```math
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\underbrace{\frac{1}{\det(A)}
\begin{bmatrix} a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix}}_{A^{-1}}
\begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
```

$`A^{-1}`$ — **Inverse of $`A`$.**

(Side note in red, bottom right: $`y = x^3`$, $`(-8) \rightarrow x\,?`$)

## Page 6

[Diagram: $`x = (x_1, x_2)^\top`$ → box $`A`$ → $`b = (b_1, b_2)^\top`$, which then feeds into box $`A^{-1}`$ → $`x = (x_1, x_2)^\top`$.]

```math
A^{-1} = \frac{\text{Adj}(A)}{\det(A)}
```

```math
\Rightarrow x = A^{-1} b.
```

Q:

1. Does $`A^{-1}`$ exist for all matrices?
2. What if $`\det(A) = 0`$?
3. Are there matrices for which $`\underline{\underline{A^{-1}}}`$ is easy to compute?

## Page 7

<u>Ex:</u>

```math
A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
```

$`\qquad`$

```math
Ax = b \Rightarrow \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}[x_1, x_2]^\top = [x_1, x_2]^\top
```


```math
\Rightarrow A^{-1} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = A. \quad \Rightarrow \text{Easy to compute } A^{-1} \Rightarrow A^{-1} = A.
```

Q: Are there more $`2 \times 2`$ matrices for which we have $`A^{-1} = A`$?


## Page 8

**Lecture - 2**
**19.09.2026.**

```math
* \quad \underbrace{\begin{bmatrix} a & b \\ c & d \end{bmatrix}}_{A}
\begin{bmatrix} x \\ y \end{bmatrix}
=
\begin{bmatrix} y \\ x \end{bmatrix}
\qquad A[(x, y)] = (y, x)
```

```math
\begin{aligned}
ax + by &= y \\
cx + dy &= x.
\end{aligned}
```

```math
\underset{A}{\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}}
\underset{X}{\begin{bmatrix} x \\ y \end{bmatrix}}
=
\underset{Y}{\begin{bmatrix} y \\ x \end{bmatrix}}
```

```math
\begin{bmatrix} e & f \\ g & h \end{bmatrix}
\begin{bmatrix} y \\ x \end{bmatrix}
=
\begin{bmatrix} x \\ y \end{bmatrix}
```

```math
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
\begin{bmatrix} y \\ x \end{bmatrix}
=
\begin{bmatrix} x \\ y \end{bmatrix}
```

```math
= \underset{A}{\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}}
\underset{A}{\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}}
\begin{bmatrix} x \\ y \end{bmatrix}
=
\begin{bmatrix} x \\ y \end{bmatrix}
```

## Page 9



```math
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
```

is a **reflection matrix** whose inverse is itself.

```math
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} x \\ x \end{bmatrix} = \begin{bmatrix} x \\ x \end{bmatrix}
\qquad\qquad
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \text{ reflects about } y = x \text{ line.}
```

```math
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
```

[Diagram: x–y axes with the line $`y = x`$ through the origin, labelled "Mirror". A point $`(x_1, y_1)^\top`$ above the line has an arrow toward the line; its mirror image $`(y_1, x_1)^\top`$ is shown below the line.]

Every matrix that reflects about line passing thro' (continued on next page)

## Page 10

the origin, will be its own inverse.

- **What is the next best option for inverse?**

```math
A = \begin{bmatrix} d_1 & 0 \\ 0 & d_2 \end{bmatrix} \rightarrow \text{Diagonal Matrix with non-zero diagonal elements}
```

```math
\Rightarrow \begin{bmatrix} d_1 & 0 \\ 0 & d_2 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} d_1 x \\ d_2 y \end{bmatrix}
```

Inverse of $`A`$:

```math
\begin{bmatrix} 1/d_1 & 0 \\ 0 & 1/d_2 \end{bmatrix}\begin{bmatrix} d_1 x \\ d_2 y \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix}
```

```math
\begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} c \\ d \end{bmatrix}
```

```math
\Rightarrow \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} c/a \\ d/b \end{bmatrix} \qquad a \neq 0,\ b \neq 0.
```

## Page 11

**Ex 2:**

```math
\begin{bmatrix} \underline{\underline{0}} & d_1 \\ d_2 & \underline{\underline{0}} \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} c \\ d \end{bmatrix} \qquad \text{PIVOT}
```

```math
\begin{aligned}
d_1 y &= c & \qquad y &= c/d_1 \\
d_2 x &= d & \qquad x &= d/d_2.
\end{aligned}
```

Det: $`-(d_1 d_2)`$

Adj:

```math
\begin{bmatrix} 0 & -d_1 \\ -d_2 & 0 \end{bmatrix}
```


- **3rd best option** $`A^{-1} = A^T`$. Given

  ```math
  A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
  ```


```math
A^T = \begin{bmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \end{bmatrix}
```

Can we have $`A`$ s.t. $`A^{-1} = A^T`$?

## Page 12

```math
\Rightarrow A^{-1}A = A^T A = I.
```

```math
\begin{bmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \end{bmatrix}
\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
=
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
```

```math
\begin{bmatrix} a_{11}^2 + a_{21}^2 & a_{11}a_{12} + a_{21}a_{22} \\ a_{12}a_{11} + a_{22}a_{21} & a_{12}^2 + a_{22}^2 \end{bmatrix}
=
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}.
```

**[Boxed]**

```math
\Rightarrow \boxed{\begin{aligned} a_{11}^2 + a_{21}^2 = a_{12}^2 + a_{22}^2 &= 1 \\ a_{11}a_{12} + a_{21}a_{22} &= 0 \end{aligned}}
```

## Page 13

$`Ax = b`$ and if inverse is defined & exists, then

```math
x = A^{-1}b.
```

However $`A^{-1}`$ is difficult to compute..

So do we have tech that gets us the value of $`x`$ without computing the inverse?

**Gaussian Elimination** & <u>Row</u> <u>Reduced</u> <u>Echelon</u> form.
$`\Downarrow`$ & Backward Substitⁿ.
Iterative

[CHECK: Gaussian elimination is labelled "Iterative"; it is normally classed as a direct (finite-step) method, not an iterative one. "Iterative" here may just mean it proceeds step by step.]

```math
\left.
\begin{aligned}
x_1 + x_2 + x_3 &= 3 \\
x_1 - x_2 + x_3 &= 1 \\
x_1 + x_2 - x_3 &= 1
\end{aligned}
\right\}
\quad \text{Solve for } x_1, x_2 \text{ \& } x_3.
```

## Page 14

```math
\begin{matrix} \text{Eq 1: } R_1 \\ \text{Eq 2: } R_2 \\ \text{Eq 3: } R_3 \end{matrix}
\begin{bmatrix} 1 & 1 & 1 \\ 1 & -1 & 1 \\ 1 & 1 & -1 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
=
\begin{bmatrix} 3 \\ 1 \\ 1 \end{bmatrix}
```

Eliminate $`x_1`$ from eqn 2.

```math
\underbrace{\left[\begin{array}{ccc|c} 1 & 1 & 1 & 3 \\ 1 & -1 & 1 & 1 \\ 1 & 1 & -1 & 1 \end{array}\right]}_{[A \,|\, b]}
\Rightarrow \text{Augmented Matrix}
```

$`R_2 \leftarrow R_2 - R_1`$

```math
= \left[\begin{array}{ccc|c} 1 & 1 & 1 & 3 \\ 0 & -2 & 0 & -2 \\ 1 & 1 & -1 & 1 \end{array}\right]
```

---


## Page 15

$`R_3 \leftarrow R_3 - R_1`$

```math
\left[\begin{array}{ccc|c}
1 & 1 & 1 & 3 \\
0 & -2 & 0 & -2 \\
0 & 0 & -2 & -2
\end{array}\right]
\;\longrightarrow\; \text{Upper triangular Matrix}
```

[Diagram: a curved arrow points from the augmented matrix to the label "Upper triangular Matrix".]

```math
\Rightarrow\quad
\begin{aligned}
x_1 + x_2 + x_3 &= 3 \\
-2x_2 &= -2 \\
-2x_3 &= -2
\end{aligned}
\qquad\Rightarrow\qquad
\begin{aligned}
x_1 &= 1 \\
x_2 &= 1 \\
x_3 &= 1
\end{aligned}
```

**Upper $`\Delta`$lar:**

```math
\begin{bmatrix}
* & * & * & * \\
0 & * & * & * \\
&   & * & * \\
&   & 0 & *
\end{bmatrix}
```

[Diagram: the region below the diagonal is circled to show it is all zeros.]

**Lower $`\Delta`$:**

```math
\begin{bmatrix}
* & 0 &   &   \\
* & * &   &   \\
* & * & * &   \\
* & * & * & *
\end{bmatrix}
```

[Diagram: the region above the diagonal is circled to show it is all zeros.]

---

## Page 16

```math
\begin{bmatrix}
* & * & * & * & * \\
0 & * &   &   &   \\
0 & 0 & * &   &   \\
0 & 0 & 0 & * &   \\
0 & 0 & 0 & 0 & *
\end{bmatrix}_{5\times 5}
```

[Diagram: a green staircase line runs under the diagonal pivots of the $`5\times5`$ matrix. Arrows mark the first column and the second row. Next to it is an empty bracketed matrix with a single diagonal stroke.]

```math
A = \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & -1 & 1 \\
1 & 1 & -1 \\
-1 & 1 & -1
\end{bmatrix}_{5\times 3}
```

$`R_2 \leftarrow R_2 - R_1`$

```math
\begin{bmatrix}
1 & 1 & 1 \\
0 & 0 & 0 \\
1 & -1 & 1 \\
1 & 1 & -1 \\
-1 & 1 & -1
\end{bmatrix}
```

[Diagram: an arrow marks row 2, which is all zeros. The 0 in position (2,2) is underlined and a curved arrow links it to the $`-1`$ below it. A bracket on the right connects row 2 with row 5.]

$`R_2 \leftrightarrow R_5`$: Swap $`R_2`$ with $`R_5`$

---

## Page 17

```math
\begin{bmatrix}
1 & 1 & 1 \\
-1 & 1 & -1 \\
1 & -1 & 1 \\
1 & 1 & -1 \\
0 & 0 & 0
\end{bmatrix}
\;\Rightarrow\;
R_2 \leftarrow R_2 + R_1:\;
\begin{bmatrix}
1 & 1 & 1 \\
0 & 2 & 0 \\
1 & -1 & 1 \\
1 & 1 & -1 \\
0 & 0 & 0
\end{bmatrix}
```

[Diagram: green arrows show each row of the left matrix carried over to the right matrix. The zero row at the bottom is circled and underlined.]

$`R_3 = R_3 - R_1`$

```math
\begin{bmatrix}
1 & 1 & 1 \\
0 & \boxed{2} & 0 \\
0 & -2 & 0 \\
1 & 1 & -1 \\
0 & 0 & 0
\end{bmatrix}
\;\longrightarrow\;
R_4 \leftarrow R_4 - R_1:\;
\begin{bmatrix}
1 & 1 & 1 \\
0 & 2 & 0 \\
0 & -2 & 0 \\
0 & 0 & -2 \\
0 & 0 & 0
\end{bmatrix}
```

(The pivot 2 is circled and the $`-2`$ below it is underlined.)

```math
R_3 \leftarrow R_3 + R_2 \;\Rightarrow\;
\begin{bmatrix}
1 & 1 & 1 \\
0 & 2 & 0 \\
0 & 0 & 0 \\
0 & 0 & -2 \\
0 & 0 & 0
\end{bmatrix}
```

(An arrow marks row 3, which is now all zeros.)

---

## Page 18

Swap $`R_3`$ & $`R_4`$:

```math
\begin{bmatrix}
1 & 1 & 1 \\
0 & 2 & 0 \\
0 & 0 & -2 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
\Bigg\}\; \text{REF.}
```

**Reduced Row Echelon form:**

(i) Every pivot element is a 1.

(ii) Every element below the pivot is 0.

(iii) Every element above the pivot is also 0.

(iv) The rows containing all elements 0 must be at the bottom.

---

## Page 19

**Reference Textbooks.**

1. Farin & Hansford $`\rightarrow`$ Practical Linear Algebra.
2. Gilbert Strang $`\rightarrow`$ Linear Algebra Ed. 5.
3. Intro to Applied Linear Algebra - Stephen Boyd.
4. Advanced Matrix Theory - NPTEL by Prof Vittal Rao
5. Linear Algebra thro Geometry - NPTEL - Ashok Rao & Arulalan Rajan

**Agenda: 22nd Sept:**

- EROW opeans [elementary row operations]
- Homogeneous Sys. of Eqns
- Vectors.

---

## Page 20

**23.09.2026 — Lecture 3.**

\* **Row operations:**

**1: $`R_i \leftrightarrow R_j`$. Row swap.**

Ex:

```math
\begin{aligned}
2x_1 + x_2 &= 3 \\
1x_1 + 2x_2 &= 3
\end{aligned}
\;\Rightarrow\;
\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 3 \\ 3 \end{bmatrix}
```

Row swap: $`R_1 \leftrightarrow R_2`$.

```math
\left[\begin{array}{cc|c} 2 & 1 & 3 \\ 1 & 2 & 3 \end{array}\right]
=
\left[\begin{array}{cc|c} 1 & 2 & 3 \\ 2 & 1 & 3 \end{array}\right]
```

[CHECK: "=" is written between the original and the row-swapped augmented matrices. They are row-equivalent, not equal.]

```math
\Rightarrow
\begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 3 \\ 3 \end{bmatrix}
\;\Rightarrow\;
\begin{aligned}
x_1 + 2x_2 &= 3 \\
2x_1 + x_2 &= 3
\end{aligned}
```

The matrix which does row swap is

```math
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \;\Rightarrow\; \text{Reflection Matrix}
```

---

## Page 21

**2. Replacing a row with non-zero multiple of the same row.**

Suppose $`[x_1, x_2]^\top`$ is a matrix, we want to replace $`x_1`$ with $`k x_1`$

```math
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} k x_1 \\ x_2 \end{bmatrix}
\;\Rightarrow\;
\begin{aligned}
a x_1 + b x_2 &= k x_1 + 0 x_2 \\
c x_1 + d x_2 &= 0 x_1 + 1 x_2
\end{aligned}
```

```math
\Rightarrow
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
=
\begin{bmatrix} k & 0 \\ 0 & 1 \end{bmatrix}
\qquad k \neq 0.
```

```math
\boxed{\begin{bmatrix} k & 0 \\ 0 & 1 \end{bmatrix}}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} k x_1 \\ x_2 \end{bmatrix}
\;;\qquad
\boxed{\begin{bmatrix} 1 & 0 \\ 0 & k \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} x_1 \\ k x_2 \end{bmatrix}}
\qquad k \neq 0.
```

(Both matrices are circled in green.)

---

## Page 22

Ex:

```math
\begin{aligned}
2x_1 + x_2 &= 3 \\
x_1 + 2x_2 &= 3
\end{aligned}
\qquad
\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 3 \\ 3 \end{bmatrix}
```

Replace $`R_1`$ with $`4R_1`$

```math
\Rightarrow
\begin{bmatrix} 8 & 4 \\ 1 & 2 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 12 \\ 3 \end{bmatrix}
```

Solution does not change.

**3: $`R_i \leftarrow R_i + k R_j`$.** $`\quad k`$: Any real no.

```math
\begin{aligned}
2x_1 + x_2 &= 3 \\
x_1 + 2x_2 &= 3
\end{aligned}
\;\Rightarrow\;
\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 3 \\ 3 \end{bmatrix}
```

$`R_2 \leftarrow R_2 + k R_1`$.

```math
\Rightarrow
\begin{bmatrix} 2 & 1 \\ 1+2k & 2+k \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 3 \\ 3+3k \end{bmatrix}
```

What does this operation do?

---

## Page 23

```math
\begin{bmatrix} 1 & 0 \\ k & 1 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} x_1 \\ x_2 + k x_1 \end{bmatrix}
```

Let $`k = 1`$

```math
\Rightarrow
\begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}
\;\Rightarrow\;
\begin{bmatrix} 0 & 1 & 1 & 0 \\ 0 & 1 & 2 & 1 \end{bmatrix}
```

[Diagram: in the $`x_1`$–$`x_2`$ plane, the blue unit square has corners $`(0,0), (1,0), (1,1), (0,1)`$. The green sheared parallelogram has corners $`(0,0), (1,1), (1,2), (0,1)`$. An upward arrow beside $`\binom{0}{1}`$ is labelled "Shearing in $`x_2`$ direction". Points $`\binom{0}{0}, \binom{1}{0}, \binom{2}{0}, \binom{3}{0}`$ are marked on the $`x_1`$ axis.]

Shearing in $`x_1`$ direc$`^n`$

[Red annotation above this heading: "Regular Font to ital[illegible]", probably "to italic".]

```math
\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}
=
\begin{bmatrix} 0 & 1 & 2 & 1 \\ 0 & 0 & 1 & 1 \end{bmatrix}
```

(The $`0`$ in position (2,1) of

```math
\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
```

is underlined in red.)

[Diagram: the blue unit square and the green sheared parallelogram with corners $`(0,0), (1,0), (2,1), (1,1)`$. A rightward arrow is labelled "Shearing along $`x_1`$ direction". Points $`\binom{0}{0}, \binom{0}{1}, \binom{1}{0}, \binom{2}{0}, \binom{3}{0}`$ are marked.]

---

## Page 24

Draw $`2x_1 + x_2 = 3`$.

```math
(1+2k)x_1 + (2+k)x_2 = 3 + 3k.
```

```math
\begin{aligned}
k = -1 &\Rightarrow -x_1 + x_2 = 0. \\
k = -2 &\Rightarrow -3x_1 = -3 \Rightarrow x_1 = 1 \\
k = 1 &\Rightarrow 3x_1 + 3x_2 = 6. \\
k = 2 &\Rightarrow 5x_1 + 4x_2 = 9 \\
k = 0 &\Rightarrow x_1 + 2x_2 = 3.
\end{aligned}
```

(Written at the top right in purple: $`x_1 + x_2 = 2`$.)

[Diagram: the $`x_1`$–$`x_2`$ plane, with both axes marked 0–6. Lines drawn: $`2x_1 + x_2 = 3`$ (green), $`-x_1 + x_2 = 0`$ (blue, labelled), the vertical line $`x_1 = 1`$ (red), $`x_1 + 2x_2 = 3`$ (blue-violet), $`3x_1+3x_2=6`$ (purple) and $`5x_1+4x_2=9`$ (orange). All the lines pass through the common point $`(1,1)`$, which is marked with a green dot.]

**With these row operations, the soln is guaranteed to remain the same.**

---

## Page 25

How do we know if a system of linear equations, $`Ax = b`$ have

a) Unique Soln &nbsp;&nbsp; or &nbsp;&nbsp; b) infinitely many solutions

given that $`Ax = b`$ has solution?

\* To figure out if $`Ax = b`$ has infinitely many solns (given that it has soln) look at $`Ax = 0`$.

Solve $`Ax = 0`$.

```math
\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 0 \\ 0 \end{bmatrix}
```

$`\Rightarrow x_1 = 0,\; x_2 = 0`$ IS <u>ALWAYS A SOLUTION</u>. $`\Rightarrow`$ TRIVIAL SOLN

```math
\left\{ \begin{pmatrix} 0 \\ 0 \end{pmatrix} \right\}.
```

```math
\begin{aligned}
a_{11} x_1 + a_{12} x_2 &= 0 \\
a_{21} x_1 + a_{22} x_2 &= 0
\end{aligned}
```

[Note: in the first equation the subscript "11" is written below the $`a`$. In the second equation the subscript of the first $`x`$ looks like $`x_\ell`$, but $`x_1`$ is clearly meant.]

[Diagram: $`[x_1, x_2]^\top \longrightarrow \boxed{[A]} \longrightarrow (0, 0)^\top`$, i.e. a block diagram where the input vector passes through $`A`$ and gives the zero vector.]

---

## Page 26

[Diagram: the "o/p" axis is drawn vertically and labelled $`y`$, and the "i/p" axis horizontally, labelled $`x`$. A single point is marked at the origin.]

Ex:

```math
\begin{aligned}
2x_1 + x_2 &= 0 \\
4x_1 + 2x_2 &= 0.
\end{aligned}
```

| $`x_1`$ | 0 | 1 | $`-1`$ | 2 | $`-2`$ | $`\cdots`$ |
|---|---|---|---|---|---|---|
| $`x_2`$ | 0 | $`-2`$ | 2 | $`-4`$ | 4 | $`\cdots`$ |

($`\{(\;)\}`$ is written above the table. An arrow extends to the right and a brace spans the solution pairs.)

If $`Ax = 0`$ has a non-trivial soln, $`x_H`$

$`x_H \neq 0`$ is a soln $`\qquad\qquad 0\,x_H = 0`$ is a soln.

$`A x_H = 0`$

Let $`k`$ be a real no

```math
k(A x_H) = k(0) = 0
```

In the above ex $`x_H = (1, -2)^\top`$ (underlined)

---

## Page 27

```math
\Rightarrow A(k x_H) = 0
```

Since $`k`$ is a real no, $`k`$ can take infinitely many values.

```math
\begin{aligned}
A x &= b \\
+\quad A(k x_H) &= 0 \\
\hline
\Rightarrow A(x + k x_H) &= b + 0 = b.
\end{aligned}
```

$`\Rightarrow`$ If $`x_H`$ is a non trivial $`Ax = 0`$, then $`Ax = b`$ WILL HAVE INFINITELY MANY SOL IF $`Ax = b`$ does have a soln.

[CHECK: "$`x_H`$ is a non trivial $`Ax=0`$" is missing a word. It presumably means "$`x_H`$ is a non-trivial solution of $`Ax = 0`$".]

**$`Ax = 0`$ is called the homogeneous system of eqns.**

---

## Page 28

① If $`A^{n\times n}`$ is a matrix with non-zero determinant, then we define the rank of $`A`$ as the number of cols. of $`A`$.

② Rank of the matrix $`A`$ is the number of pivot cols in the RREF of $`A`$.

③ Rank of the matrix $`A`$ is the dimension of the largest possible square matrix for which the determinant $`\neq 0`$.

[CHECK: in ③ the standard statement is "largest square **submatrix** of $`A`$" with non-zero determinant, i.e. a non-zero minor. As written, "square matrix" is ambiguous.]

For ex:

```math
A = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}_{2\times 2}
\qquad \det(A) = 0 \Rightarrow \text{Rank} \neq 2.
```

```math
\Rightarrow \text{Rank of } A = 1. \qquad [2]_{1\times 1}
```

(The bottom row of $`A`$ is underlined with a wavy line. $`[2]_{1\times1}`$ is the $`1\times1`$ submatrix with non-zero determinant.)

---



## Page 29

```math
A = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix} \qquad \text{Rank} = 0.
```

```math
A = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix} \qquad \det = 0 \Rightarrow \text{Rank} = 1.
```

[CHECK: $`\det = 0`$ alone only shows rank $`< 2`$. Rank $`= 1`$ is correct here because the matrix has a non-zero entry.]

**Evariste Galois** $`\underline{\underline{21}}`$

<u>Galois Theory</u> $`\rightarrow`$ $`GF(p^k)`$. (with a small mark under $`p`$)

[CHECK: Galois died at age 20 (1811–1832). "21" may be a different reference, for example a lecture/slide number.]

---

## Page 30

*(Date, top right: 26.09.2026.)*

**Vectors.**

vector: Direction & Magnitude

2D: $`(x, y)`$ or $`(x, y)^\top`$

Ex: $`(2, 3)^\top`$ is a 2-component vector.

Ordered pair of numbers.

**Vector Space over a Set of real nos**

2 Component Vector $`(x, y)^\top`$, $`\quad x \in \mathbb{R}`$, $`\; y \in \mathbb{R}`$.

[Diagram: $`x`$–$`y`$ axes. A vector from the origin to the point $`(x_1, y_1)^\top`$. Its components are marked as $`x_1`$ on the $`x`$-axis and $`y_1`$ on the $`y`$-axis.]

---

## Page 31

Basket $`\mathcal{V}`$ contains some elements s.t. if $`\vec{u}`$ & $`\vec{v}`$ are elements of $`\mathcal{V}`$, then we define **vector addition** as

*(side note)* Let $`\vec{u} = (u_1, u_2)^\top`$ & $`\vec{v} = (v_1, v_2)^\top`$

```math
\vec{u} + \vec{v} = \begin{pmatrix} u_1 \\ u_2 \end{pmatrix} + \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \end{pmatrix} = \begin{pmatrix} w_1 \\ w_2 \end{pmatrix} = \vec{w}
```

If $`\vec{u} + \vec{v}`$ is also an element of $`\mathcal{V}`$, then we say $`\mathcal{V}`$ is **closed under vector addition**.

2. We define **scalar multiplication** of a vector as

---

## Page 32

for some real number $`\alpha`$,

```math
\alpha \vec{u} = \alpha \begin{pmatrix} u_1 \\ u_2 \end{pmatrix} = \begin{pmatrix} \alpha \cdot u_1 \\ \alpha \cdot u_2 \end{pmatrix} = \vec{z} = \begin{pmatrix} z_1 \\ z_2 \end{pmatrix}
```

```math
5 \begin{pmatrix} 2 \\ 3 \end{pmatrix} = \begin{pmatrix} 5(2) \\ 5(3) \end{pmatrix} = \begin{pmatrix} 10 \\ 15 \end{pmatrix}
```

If for some scalar $`\alpha`$, real & a vector $`\vec{u}`$ an element of $`\mathcal{V}`$, $`\alpha(\vec{u}) \in \mathcal{V}`$ then we say $`\mathcal{V}`$ is **closed under scalar multiplication**.

③ $`(0, 0)^\top \in \mathcal{V}`$. $`\;\hookrightarrow`$ **zero vector**

---

## Page 33

$`\mathcal{V}(+, \cdot)`$ over a set of real numbers is a **vector space** if

a) $`\mathcal{V}`$ is closed under vector addition $`\quad`$ (side example: $`(2, 3)^\top + (4, 5)^\top = (6, 8)^\top`$)

b) closed under scalar multiplication

& c) $`\mathcal{V}`$ contains an element called the **zero vector**.

[Diagram: A number line labelled $`\mathbb{R}`$ with $`0`$ marked, and arrows pointing both ways.]

$`xy`$ plane – 2D. $`\hookrightarrow \mathbb{R} \times \mathbb{R} = \mathbb{R}^2`$, $`\quad (x_1, y_1)^\top`$

[Diagram: The $`xy`$-plane. Both axes are labelled $`\mathbb{R}`$, and the origin is marked $`(0, 0)^\top`$. Vectors drawn from the origin: $`(x_1, y_1)^\top`$ (labelled as $`(2, 3)^\top`$), $`(x_2, y_2)^\top`$ (labelled as $`(4, 1)^\top`$), and their sum along the diagonal, labelled $`(x_1, y_1)^\top`$ [partly illegible]. Together they show the parallelogram rule for addition.]

---

## Page 34

**Ex Vector Spaces:**

\* **Ex 1:** Set of real numbers $`\mathbb{R}`$ is a vector space.

\* **Ex 2:** $`\mathbb{R}^2`$ is a VS

```math
3\begin{pmatrix} 2 \\ 3 \end{pmatrix} = \begin{pmatrix} 6 \\ 9 \end{pmatrix}, \qquad -3\begin{pmatrix} 2 \\ 3 \end{pmatrix} = \begin{pmatrix} -6 \\ -9 \end{pmatrix}
```

```math
\begin{pmatrix} 4 \\ 2 \end{pmatrix} + \begin{pmatrix} 2 \\ 3 \end{pmatrix} = \begin{pmatrix} 6 \\ 5 \end{pmatrix}
```

[Diagram: Axes with the origin $`(0, 0)^\top`$. Vector $`(2, 3)^\top`$ is drawn with dashed guides at $`x = 2`$ and $`y = 3`$. Vector $`(4, 2)^\top`$ is also drawn. Translated copies of each complete a parallelogram, and the diagonal goes to $`(6, 5)^\top`$, with a dashed guide at $`x = 6`$. Edges are tick-marked.]

---

## Page 35

**Ex: 3** $`\quad \underline{S_1} = \left\{ (x_1, x_1)^\top,\; x_1 \in \mathbb{R} \right\}`$.

```math
\vec{u} = \begin{pmatrix} u_1 \\ u_1 \end{pmatrix} \qquad \vec{u} = \begin{pmatrix} v_1 \\ v_1 \end{pmatrix}
```

[CHECK: The second vector is labelled $`\vec{u}`$ but its components are $`v_1`$. It is presumably meant to be $`\vec{v}`$, as used in the next line.]

```math
\vec{u} + \vec{v} = \begin{pmatrix} u_1 \\ u_1 \end{pmatrix} + \begin{pmatrix} v_1 \\ v_1 \end{pmatrix} = \begin{pmatrix} u_1 + v_1 \\ u_1 + v_1 \end{pmatrix} \in S_1.
```

$`S_1`$ is closed under vector addition.

Let $`k`$ be some real number

```math
k\vec{u} = k\begin{pmatrix} u_1 \\ u_1 \end{pmatrix} = \begin{pmatrix} ku_1 \\ ku_1 \end{pmatrix} \in S_1
```

$`\Rightarrow S_1`$ is closed under scalar multiplication.

[Diagram: $`\mathbb{R}^2`$ axes with two lines through the origin. One is the line $`y = x`$ (blue, ticked ✓) with points marked along it. The other is a steeper line (purple). The origin is highlighted.]

---

## Page 36

Is $`(0, 0)^\top \in S_1`$? Yes.

$`\Rightarrow S_1`$ is a vector space.

**Ex 4:** $`\quad S_2 = \left\{ (x_1, 0)^\top,\; x_1 \in \mathbb{R} \right\}`$ is a VS.

**Ex 5:** $`\quad S_3 = \left\{ (0, x_2)^\top,\; x_2 \in \mathbb{R} \right\}`$ is a VS.

**Ex 6:** $`\quad S_4 = \left\{ (x_1, k x_1)^\top,\; x_1 \in \mathbb{R},\; k \in \mathbb{R} \right\}`$ is a VS. $`\quad`$ (e.g. $`(x_1, 2x_1)^\top`$; a small "2" is written under the $`k`$)

[CHECK: As written, with both $`x_1`$ and $`k`$ ranging over $`\mathbb{R}`$, $`S_4`$ is the union of all non-vertical lines through the origin. That union is not closed under addition. The intended meaning is presumably a **fixed** $`k`$ (e.g. $`k = 2`$), which gives a single line $`y = kx`$. The conclusion below also supports this reading.]

$`\Rightarrow`$ **Every line passing through the origin is a vector space.**

---

## Page 37

**Ex 7:** $`\quad S_5 = \left\{ (x_1, 3)^\top,\; x_1 \in \mathbb{R} \right\}`$ is **not** a VS (underlined).

**Ex 8:** $`\quad S_6 = \left\{ (0, 0)^\top \right\}`$ is a vector space.

**Ex 9:** $`\quad \underline{\mathcal{M}^{2\times 2}}`$: Set of all $`2\times 2`$ real matrices. $`\quad`$ (margin note: $`(x, y)`$)

```math
A = \begin{bmatrix} a_1 & a_2 \\ a_3 & a_4 \end{bmatrix} \qquad B = \begin{bmatrix} b_1 & b_2 \\ b_3 & b_4 \end{bmatrix}
```

```math
A + B = \begin{bmatrix} a_1 + b_1 & a_2 + b_2 \\ a_3 + b_3 & a_4 + b_4 \end{bmatrix} \in \mathcal{M}^{2\times 2}
```

$`\mathcal{M}^{2\times 2}`$ is closed under vector addition

---

## Page 38

Let $`k \in \mathbb{R}`$

```math
kA = k\begin{bmatrix} a_1 & a_2 \\ a_3 & a_4 \end{bmatrix} = \begin{bmatrix} ka_1 & ka_2 \\ ka_3 & ka_4 \end{bmatrix} \in \mathcal{M}^{2\times 2}
```

$`\mathcal{M}^{2\times 2}`$ is closed under scalar multiplication

Is

```math
\begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix} \in \mathcal{M}^{2\times 2}
```

? YES.

$`\therefore \mathcal{M}^{2\times 2}`$ is a vector space.

---

## Page 39

**Vector Space** is a collection of elements closed under addition & scalar multiplication and contains the zero element.

Every element of a VS is called a **vector**.

Ordered $`n`$-tuple has mag & direction.

---

## Page 40

*(Date, top right: 30.09.2026.)*

Recall Ex: 3, …, 8.

**Q:** Does the set $`S_1`$ in Ex 3 contain all the points in $`\mathbb{R}^2`$? $`\quad`$ No.

However $`S_1`$ is a subset of $`\mathbb{R}^2`$, closed under VA, SM & has zero vector $`(0, 0)^\top`$.

$`S_1`$ is called a **vector subspace** of $`\mathbb{R}^2`$.

All the sets from Ex 3 to Ex 8 are all subsets of $`\mathbb{R}^2`$ which are vector spaces by themselves.

$`\Rightarrow`$ All these sets are vector subspaces of $`\mathbb{R}^2`$.

[CHECK: Ex 7 ($`S_5`$) is in the range Ex 3–8, but it was stated to be **not** a VS. So not all of Ex 3–8 are subspaces.]

---

## Page 41

**Defn:** Any subset of a vector space, which by itself is a vector space is called a **vector subspace**.

**Geometry of Subspaces of $`\mathbb{R}^2`$:**

\* $`\left\{ (0, 0)^\top \right\}`$ is a subspace of $`\mathbb{R}^2`$.

\* Any line passing thro' the origin is a vector subspace of $`\mathbb{R}^2`$

\* $`\mathbb{R}^2`$ is a subspace of $`\mathbb{R}^2`$. $`\rightarrow`$ Every plane passing thro' the origin is a subsp of a larger vector space.

---

## Page 42

\* **How do we generate vector spaces?**

For ex let $`\mathcal{V} = \mathbb{R}^2`$

How do we generali [sic, likely "generate"] all the vectors in $`\mathbb{R}^2`$?

Choose 2 vectors $`\vec{u}`$ & $`\vec{v}`$ s.t. $`\vec{u}, \vec{v}`$ are in 2 different directions. $`\quad \vec{u} = (u_1, u_2)^\top`$, $`\; \vec{v} = (v_1, v_2)^\top`$

For $`\alpha, \beta`$ real,

```math
\alpha\vec{u} + \beta\vec{v} \Rightarrow \textbf{Linear combination of } \vec{u} \text{ \& } \vec{v}.
```

```math
\alpha\begin{pmatrix} u_1 \\ u_2 \end{pmatrix} + \beta\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} \alpha u_1 + \beta v_1 \\ \alpha u_2 + \beta v_2 \end{pmatrix} = \begin{pmatrix} \cdot \\ \cdot \end{pmatrix}
```

```math
\vec{u} = \begin{pmatrix} 2 \\ 5 \end{pmatrix} \qquad \vec{v} = \begin{pmatrix} 1 \\ 5 \end{pmatrix}
```

*(Top-right annotation:)* $`\alpha\vec{u} + \beta\vec{v}`$, with ticks over $`\vec{u}`$ and $`\vec{v}`$. Arrows point up to them from the words "[illegible — possibly 'palinal' / 'Malinal']" written under $`\alpha\vec{u}`$ and $`\beta\vec{v}`$ respectively.

---



## Page 43

**Claim:** $`\alpha[2, 5]^\top + \beta[1, 5]^\top`$ for different $`\alpha`$ & $`\beta`$ results in entire $`\mathbb{R}^2`$.

```math
\Rightarrow \begin{bmatrix}2\alpha+\beta\\5\alpha+5\beta\end{bmatrix} = \underbrace{\begin{bmatrix}2&1\\5&5\end{bmatrix}}_{A}\underbrace{\begin{bmatrix}\alpha\\\beta\end{bmatrix}}_{\vec{x}} = \begin{bmatrix}w_1\\w_2\end{bmatrix}
```

$`A\vec{x}`$ $`\Rightarrow \text{Det}(A) \neq 0`$

$`\Rightarrow A^{-1}`$ exists.

(Side note, left margin:) $`A\vec{x} = \vec{b}`$, $`\;\vec{x} = A^{-1}\vec{b}`$

For every $`[\alpha, \beta]^\top`$,

```math
\begin{bmatrix}2&1\\5&5\end{bmatrix}[\alpha, \beta]^\top
```

gives different vectors.

$`\vec{u} = [1, 0]^\top`$, $`\vec{v} = [0, 1]^\top`$ $`\Rightarrow x_1[1, 0]^\top + x_2[0, 1]^\top`$ is the set of all possible linear combination for different $`[x_1, x_2]^\top`$.

## Page 44

[Diagram: $`x_1`$–$`x_2`$ axes with ticks 0–5 on each axis. Two arrows from the origin: a green arrow to $`[1, 5]^\top`$ and a purple arrow to $`[2, 5]^\top`$.]

Suppose $`\vec{u} = [1, 5]^\top`$, $`\vec{v} = [2, 5]^\top`$, $`\vec{w} = [3, 5]^\top`$

```math
\alpha\vec{u} + \beta\vec{v} + \gamma\vec{w}
```

\* **Defn:** A set of vectors $`\{\vec{u}, \vec{v}, \vec{w}\}`$ is said to be **linearly indep** if and only if, for scalars $`c_1, c_2, c_3`$ real,

```math
c_1\vec{u} + c_2\vec{v} + c_3\vec{w} = \vec{0} \iff \boxed{c_1 = 0,\ c_2 = 0\ \&\ c_3 = 0}
```

## Page 45

$`\vec{u} = [1, 0]^\top`$ ✓, $`\vec{v} = [0, 1]^\top`$ ✓, $`\vec{w} = [2, 3]^\top`$ ✓

```math
c_1\begin{bmatrix}1\\0\end{bmatrix} + c_2\begin{bmatrix}0\\1\end{bmatrix} + c_3\begin{bmatrix}2\\3\end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix}
```

Look at that l.c which results in $`[0, 0]^\top`$

- $`c_1 = 0, c_2 = 0, c_3 = 0 \Rightarrow c_1[1, 0]^\top + c_2[0, 1]^\top + c_3[2, 3]^\top = [0, 0]^\top`$
- $`c_1 = 2, c_2 = 3, c_3 = -1 \Rightarrow 2[1, 0]^\top + 3[0, 1]^\top - 1[2, 3]^\top = [0, 0]^\top`$

- $`\Rightarrow [1, 5]^\top, [2, 5]^\top \Rightarrow \mathbb{R}^2`$ (this pair is circled)
- $`\Rightarrow [1, 5]^\top, [3, 5]^\top = \mathbb{R}^2`$
- $`\Rightarrow [1, 0]^\top, [0, 1]^\top \Rightarrow \mathbb{R}^2`$
- $`\Rightarrow [2, 5]^\top, [3, 5]^\top = \mathbb{R}^2`$

## Page 46

**Basis:** It is the set of linearly indep vectors whose all possible linear combinations generate an entire vector space.

**Ex 1:** $`\vec{u} = \left\{[1, 1]^\top\right\}`$ generates the subspace $`\left\{x_1[1, 1]^\top\right\}`$

Basis for the vs $`\left\{[x_1, x_1]^\top\right\}`$. $`\quad \left\{[x_1, x_1]^\top\right\}`$

**Ex 2:** $`V = \left\{[x_1, 2x_1]^\top, x_1 \in \mathbb{R}\right\}`$ $`\quad`$ Basis: $`\left\{[1, 2]^\top\right\}`$.

**Ex 3:** $`V = \left\{[-3x_1, x_1]^\top, x_1 \in \mathbb{R}\right\}`$ $`\quad`$ Basis: $`\left\{[-3, 1]^\top\right\}`$.

## Page 47

**Comments:**

1. A set containing only one nonzero vector is a linearly indep set.
   Q: Is $`\left\{[1, 0]^\top\right\}`$ a l.i set? Yes.

2. A set that has $`n`$ linearly indep vectors is a basis for an $`n`$-dim vector space.
3. Any set that contains the zero vector is a linearly dep set.
4. Every vector space has a basis, in fact infinitely many basis — However all the bases will have the same number of l.i vectors.

[CHECK: Comment 2 implicitly assumes the $`n`$ l.i. vectors lie in that $`n`$-dim space; as written it omits that condition.]

## Page 48

For ex: $`\mathbb{R}^2`$ is vector space

```math
B_1 = \left\{\begin{bmatrix}1\\0\end{bmatrix}, \begin{bmatrix}0\\1\end{bmatrix}\right\} \qquad B_2 = \left\{\begin{bmatrix}2\\1\end{bmatrix}, \begin{bmatrix}1\\2\end{bmatrix}\right\}
```

```math
B_3 = \left\{\begin{bmatrix}1\\4\end{bmatrix}, \begin{bmatrix}-1\\2\end{bmatrix}\right\} \qquad B_4 = \left\{\begin{bmatrix}-2\\-3\end{bmatrix}, \begin{bmatrix}1\\-4\end{bmatrix}\right\}
```

All these bases have exactly 2 l.i vectors. $`\Rightarrow`$ **The number of vectors in any basis is called the DIMENSION of the vector space.**

## Page 49

5. Let $`V = \left\{[0, 0]^\top\right\} \rightarrow`$ vs.
   Basis: ?

*03.10.2026.*

\* Basis for $`V = \left\{[0, 0]^\top\right\}`$ ?

$`V`$ has only one element $`[0, 0]^\top`$. Thus $`V`$ is a linearly dep. set. $`\therefore [0, 0]^\top`$ cannot be a basis for $`V`$. $`\Rightarrow`$ Basis $`= \{\{\ \}\}`$ $`\rightarrow \left\{[0, 0]^\top\right\}`$ is a 0-D subspace.

[CHECK: The basis of the zero subspace is the empty set $`\{\}`$ (i.e. $`\emptyset`$); $`\{\{\}\}`$ as written denotes a set containing the empty set.]

## Page 50

\* **Recall:** Basis for a $`k`$-dim VS is a set of $`k`$ linearly indep vectors.
\# of vectors in any basis is called the Dimension of the vector space.

\* Let

```math
A = \begin{bmatrix}1&1\\1&-1\\2&1\end{bmatrix} \qquad x = \begin{bmatrix}x_1\\x_2\end{bmatrix} \qquad b = \begin{bmatrix}b_1\\b_2\\b_3\end{bmatrix}
```

```math
Ax = b \Rightarrow \begin{bmatrix}1&1\\1&-1\\2&1\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix} = \begin{bmatrix}b_1\\b_2\\b_3\end{bmatrix}
```

```math
= \begin{aligned} x_1 + x_2 &= b_1\\ x_1 - x_2 &= b_2\\ 2x_1 + x_2 &= b_3 \end{aligned} \quad\Rightarrow\quad x_1\begin{bmatrix}1\\1\\2\end{bmatrix} + x_2\begin{bmatrix}1\\-1\\1\end{bmatrix} = \begin{bmatrix}b_1\\b_2\\b_3\end{bmatrix}
```

## Page 51

(i) Find the soln to $`Ax = \vec{0}`$.

```math
\begin{bmatrix}1&1\\1&-1\\2&1\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix} = \begin{bmatrix}0\\0\\0\end{bmatrix} \Rightarrow \begin{aligned}x_1 &= 0\\ x_2 &= 0\end{aligned}
```

```math
\begin{aligned} x_1 + x_2 &= 0\\ x_1 - x_2 &= 0\\ 2x_1 + x_2 &= 0 \end{aligned} \quad\Rightarrow \text{Soln to } Ax = 0 \Rightarrow x = \left\{\begin{bmatrix}0\\0\end{bmatrix}\right\}
```

$`\Rightarrow`$ Look at set of solns to $`A\vec{x} = \vec{0}`$

```math
A^{m\times n}x^{n\times 1} = 0^{m\times 1}
```

$`\Rightarrow`$ Set of solutions, $`\underline{x^{n\times 1}}`$, to $`Ax = 0`$ forms a vector subspace $`\Rightarrow`$ The set of solns to $`A\vec{x} = \vec{0}`$, i.e., $`x^{n\times 1}`$, is called the **NULL SPACE (A)** / **KERNEL (A)**.

## Page 52

(2)

```math
\underbrace{\begin{bmatrix}1&1\\1&1\\1&1\end{bmatrix}}_{A}\begin{bmatrix}x_1\\x_2\end{bmatrix} = \begin{bmatrix}0\\0\\0\end{bmatrix} \quad A\vec{x} = \vec{0}
```
(with arrows labelling the parts as $`A`$, $`\vec{x}`$, $`= \vec{0}`$)

```math
\begin{aligned} x_1 + x_2 &= 0\\ x_1 + x_2 &= 0\\ x_1 + x_2 &= 0 \end{aligned}
```

$`x_1 = 0,\ 1,\ -1,\ 2,\ \dots`$
$`x_2 = 0,\ -1,\ 1,\ -2,\ \dots`$

$`\Rightarrow \left\{[x_1, -x_1]^\top\right\} \Rightarrow`$

Null sp(A): 1D Subsp. of $`\mathbb{R}^2`$.
Nullity: 1.

```math
A^{m\times n}\,x^{n\times 1} = b^{m\times 1} \qquad A \in \mathbb{R}^{m\times n}
```

```math
x \in \mathbb{R}^n \qquad b \in \mathbb{R}^m
```

Soln to $`Ax = 0 \Rightarrow x \in \mathbb{R}^n`$

For ex $`A^{3\times 2}x^{2\times 1} = b^{3\times 1}`$.
Soln to $`Ax = 0 \Rightarrow x \in \mathbb{R}^2`$, $`[x_1, x_2]^\top`$

## Page 53

```math
A = \begin{bmatrix}0&0&0\\0&0&0\\0&0&0\\0&0&0\\0&0&0\end{bmatrix} \quad x = \begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix} = \begin{bmatrix}0\\0\\0\\0\\0\end{bmatrix}
```

[CHECK: As written, "$`x = \ldots = 0_{5\times1}`$" — the intended statement is $`Ax = 0`$ (the $`A`$ and $`x`$ are written side by side, with $`=`$ before the zero vector).]

$`x \in \mathbb{R}^3`$. $`\Rightarrow`$ Entire $`\mathbb{R}^3`$ is the soln to $`A\vec{x} = \vec{0}`$

Null space: $`\mathbb{R}^3`$ $`\qquad`$ Dim of Null sp(A) = 3.

**The dimension of the null space of A is called the NULLITY of A.**

Ex: $`\mathbb{R}^5`$, NS(A) $`= \mathbb{R}^5`$, Nullity(A) = 5.

```math
\begin{bmatrix}0&0\\0&0\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix} \qquad \text{NS(A)}: \mathbb{R}^2,\ \text{Dim}: 2.
```

Nullity tells us the number of redundant features.

## Page 54

$`A^{1000\times 100}`$ $`\quad`$ nullity = 37 $`\Rightarrow`$ Out of 100, 37 variables are redundant.

[CHECK: Nullity 37 means the solution space has dimension 37 (rank 63), i.e. there are 37 independent linear dependencies among the columns; it does not single out 37 specific variables as redundant. Kept as an informal interpretation.]

```math
\underset{2\times2}{\begin{bmatrix}1&1\\1&1\end{bmatrix}}\underset{2\times1}{\begin{bmatrix}x_1\\x_2\end{bmatrix}} = \underset{2\times1}{\begin{bmatrix}0\\0\end{bmatrix}} \qquad \begin{aligned} x_1 + x_2 &= 0\\ \Rightarrow x_2 &= -x_1 \end{aligned}
```

Null Space: $`\left\{k[1, -1]^\top\right\}`$ $`\quad`$ 1D subsp. of $`\mathbb{R}^2`$

$`\mathbb{R}^2`$. $`\quad \left\{[k, -k]^\top\right\}`$, $`\quad \left\{k[1, -1]^\top\right\}`$

[Diagram: $`x_1`$–$`x_2`$ axes with the line $`x_2 = -x_1`$ through the origin (top-left to bottom-right); an arrow along the line points to $`[1, -1]^\top`$, labelled "→ 1D Subs".]

## Page 55

```math
\begin{bmatrix}1&1&2\\1&0&3\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix}
```

$`R_2 \leftarrow R_2 - R_1`$

```math
\begin{bmatrix}1&1&2\\0&-1&1\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix}
```

From 2nd row:
$`\Rightarrow -x_2 + x_3 \Rightarrow x_2 = x_3`$. Let $`x_3 = t`$, $`t`$ real
$`\therefore x_2 = t`$

[CHECK: "$`-x_2 + x_3`$" is missing "$`= 0`$"; the conclusion $`x_2 = x_3`$ is correct.]

From 1st row: $`x_1 + x_2 + 2x_3 = 0`$
$`\Rightarrow x_1 + t + 2t = 0 \Rightarrow x_1 = -3t`$.

```math
\therefore \text{Soln}: \left\{\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix} = \begin{bmatrix}-3t\\t\\t\end{bmatrix} = t\begin{bmatrix}-3\\1\\1\end{bmatrix}\right\}.
```

$`\Rightarrow`$ Null space: 1D sub. of $`\mathbb{R}^3`$ $`= \left\{t[-3, 1, 1]^\top\right\}`$

(Margin note, underlined:) *October 5/2026 — Class from 8pm – 9:30pm*

---


