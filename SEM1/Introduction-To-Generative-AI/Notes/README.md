# Introduction to Generative AI: Notes

**M.Tech Sem 1** · Instructor: *(add when known)*
**Where the course is heading:** AI/ML/DL foundations → neural networks → training (backprop, optimisers) → sequence models & attention → Transformers & LLMs → generative models (VAEs, GANs, diffusion) → prompting, fine-tuning, RLHF

## How each note is organised
Big picture → intuition & real-world examples → theory & maths → worked numerical examples (verified in code) → 🔴 advanced / research bridge → ⚠️ common confusions → 📝 exam/interview questions with answers → 🧾 cheat sheet → 📚 curated links.

Levels: 🟢 basic · 🟡 intermediate · 🔴 advanced / beyond syllabus.

## Lecture tracker

| # | Note | Slides | Sources | Status |
|---|---|---|---|---|
| 01 | [Foundations: AI, ML, DL & the Formal Learning Setup](01-AI-ML-DL-Foundations.md) | 1–17 | `Intro_to_GenAI.pdf` | 🟡 Waiting for transcript |
| 02 | [Neural Networks: Why, What & How They Compute](02-Neural-Networks-Fundamentals.md) | 18–26 | `Intro_to_GenAI.pdf` | 🟡 Waiting for transcript |
| 03 | Training neural networks: backpropagation, optimisers (expected next) | — | — | ⏳ |

**Notebook:** [`code/neural_networks_from_scratch.ipynb`](code/neural_networks_from_scratch.ipynb) covers the forward pass by hand, XOR (proof + hand-built solution), **a network trained from scratch with backprop (NumPy only)**, a gradient check, softmax/temperature, and parameter counting.
**Figures:** [`code/make_figures.py`](code/make_figures.py) → `images/`

## Cross-course links

| Topic here | Also covered in |
|---|---|
| ML paradigms, ChatGPT's SSL → SFT → RLHF | [MLP Note 01](../../Machine-Learning-Paradigms/Notes/01-Introduction-to-ML-Paradigms.md) |
| Loss, gradient descent, MSE | [MLP Note 03](../../Machine-Learning-Paradigms/Notes/03-Supervised-Learning-Regression.md) |
| $W\mathbf x$ as a linear combination, rank, feature space | [Applied Math Notes 03–05](../../Applied-Math-For-AI-DS/Notes/README.md) |

## ⚠️ Notation notes on the slides

| Slide | Issue | Reading used in the notes |
|---|---|---|
| 23 | Text says "$W_{ji}$ connects input $j$ to neuron $i$", formula uses $W_{ij}x_j$ | $W_{ij}$ = from input $j$ to neuron $i$ (row = destination), so $\mathbf a = W\mathbf x$ |
| 26 | $W_L \in \mathbb R^{n\times k}$ | $k\times n$ under the column-vector convention $\mathbf a = W\mathbf h$ |
| 22 | $\hat y = O(W_3h_2+b_3$ (missing ")") | Typo |

## Running the code
```bash
pip install numpy matplotlib scikit-learn jupyter
cd code
python make_figures.py
jupyter notebook neural_networks_from_scratch.ipynb
```
