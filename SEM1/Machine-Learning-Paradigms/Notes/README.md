# Machine Learning Paradigms

> DS602 · Dr. Sunil Saumya · IIIT Dharwad · Semester 1

## Notes

| # | Topic | Lecture | Notebooks |
|:-:|:--|:--|:--|
| 01 | [Introduction to ML Paradigms](01-Introduction-to-ML-Paradigms.md) | 1 Oct | [paradigms hands-on](code/01_paradigms_hands_on.ipynb) |
| 02 | [The ML Pipeline, Hands-On](02-ML-Pipeline-Hands-On.md) | 3 Oct | [class demo](code/02_ml_pipeline_hands_on.ipynb) · [deep dive](code/02_pipeline_deep_dive.ipynb) |
| 03 | [Regression & Gradient Descent](03-Supervised-Learning-Regression.md) | Week 2 | [basics](code/03_linear_regression.ipynb) · [deep dive](code/03_regression_deep_dive.ipynb) |
| 04 | [Classification: Logistic Regression & Metrics](04-Supervised-Learning-Classification.md) | 15 Oct | [deep dive](code/04_classification_deep_dive.ipynb) |

Each note has worked examples, a bank of 25+ practice problems with full solutions, and real-world case studies.

**Coming up:** Clustering · Semi-supervised & active learning · Self-supervised learning · Reinforcement learning · Transfer learning · Generative learning · Few-shot learning · Continual learning · Multimodal learning · Meta-learning

## Assessment

| Component | Weight |
|:--|--:|
| Attendance | 10% |
| Class notes (within 1 day) | 10% |
| Project | 15% |
| Assignments | 20% |
| Quiz (week 4) | 10% |
| Mid-sem (weeks 1–7) | 15% |
| End-sem (weeks 1–14) | 20% |

## Corrections to the course material

- **Regression deck, OLS worked example:** the slide gives `Ŷ = 4.41X − 18.00`. The correct least-squares line is **`Ŷ = 2.74X − 7.59`** (Σ dx·dy = 29.6, Σ dx² = 10.8), verified three ways in the notebook. [Details](03-Supervised-Learning-Regression.md#6-worked-example--slide-correction-)
- **Regression deck, multivariate gradient descent:** two update rules use `w₀ + w₁x⁽ⁱ⁾`; they should use the full prediction `wᵀx⁽ⁱ⁾`.

- **Classification deck:** slide 14 says P(y=1) = 0.9 at word count 17; with the slide's own weights it is **0.871**. The weights themselves are not unique (the data are perfectly separable, so no finite best fit exists). Slide 12 calls the sigmoid the "logit"; the logit is its inverse. Slide 19 is titled "3D data" but uses 2 features with squared terms. F1 for the spam example is 60/131 = **0.458**, not 0.457. [Details](04-Supervised-Learning-Classification.md#21--slide-corrections)

## Textbooks

Duda & Hart, *Pattern Classification* · Bishop, *Pattern Recognition and Machine Learning* · Yegnanarayana, *Artificial Neural Networks* · Géron, *Hands-On Machine Learning*
