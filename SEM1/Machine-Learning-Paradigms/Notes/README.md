# Machine Learning Paradigms (DS602): Notes

**Instructor:** Dr. Sunil Saumya · IIIT Dharwad · M.Tech Sem 1 (Sep 2026 – )
**Textbooks:** Duda & Hart · Bishop (PRML) · Yegnanarayana · Géron (Hands-On ML) · scikit-learn docs

## How each note is organised
Big picture → intuition → real-world examples → theory & maths (step by step) → worked numerical example → code → 🔴 advanced/research angle → 🎓 professor emphasised → ⚠️ common confusions → 📝 exam questions (with answers) → 🧾 cheat sheet → 📚 curated links.

Levels: 🟢 basic · 🟡 intermediate · 🔴 advanced / beyond syllabus.

## Lecture tracker

| # | Note | Week | Date | Sources | Notebook | Status |
|---|---|---|---|---|---|---|
| 01 | [Introduction to ML Paradigms](01-Introduction-to-ML-Paradigms.md) | 1 | 1 Oct 2026 | PPT + transcript | — | ✅ |
| 02 | [ML Pipeline, Hands-On (Sentiment + Iris)](02-ML-Pipeline-Hands-On.md) | 1 | 3 Oct 2026 | Transcript + Colab demo | [02_ml_pipeline_hands_on.ipynb](code/02_ml_pipeline_hands_on.ipynb) | ✅ |
| 03 | [Supervised Learning: Regression](03-Supervised-Learning-Regression.md) | 2 | deck dated 3 Oct | PPT only | [03_linear_regression.ipynb](code/03_linear_regression.ipynb) | 🟡 Waiting for lecture transcript |
| 04 | Classification (logistic regression, metrics) | 2–3 | — | — | — | ⏳ |
| 05 | Unsupervised: K-means, hierarchical (image segmentation) | 3 | — | — | — | ⏳ |
| 06 | Semi-supervised & Active learning | 4 | — | — | — | ⏳ |
| 07 | Self-supervised: contrastive learning (SimCLR) | 5 | — | — | — | ⏳ |
| 08 | Reinforcement learning: Q-learning, Gym | 6 | — | — | — | ⏳ |
| 09 | Transfer learning: fine-tuning ResNet | 7 | — | — | — | ⏳ |
| 10 | Generative learning: Transformers, LM | 8 | — | — | — | ⏳ |
| 11 | Zero/one/few-shot: Siamese networks | 9 | — | — | — | ⏳ |
| 12 | Continual learning: progressive NNs | 10 | — | — | — | ⏳ |
| 13 | Multimodal learning: cross-modal attention, VQA | 11 | — | — | — | ⏳ |
| 14 | Meta-learning: MAML | 12 | — | — | — | ⏳ |

## ⚠️ Errata found in course material

| Where | Issue | Correct |
|---|---|---|
| Regression deck, OLS worked example | $\hat Y = 4.41X - 18.00$ | $\hat Y = 2.74X - 7.59$ (Σ(dx·dy) = 29.6, Σdx² = 10.8). See [Note 03 §6](03-Supervised-Learning-Regression.md#6-worked-example--slide-correction-) |
| Regression deck, multivariate GD | Two update lines use $(w_0 + w_1x^{(i)} - y^{(i)})$ | Should be $(\mathbf w^\top\mathbf x^{(i)} - y^{(i)})$ |
| Regression deck, word counts | "Absolutely love it!…" = 8, "Amazing value…" = 7 | Actual whitespace counts are 7 and 6 (notes keep the slide values) |

## Exam calendar (from course plan)

- **Quiz (10%)**: week 4, theory and algorithm fundamentals
- **Mid-sem (15%)**: weeks 1–7
- **End-sem (20%)**: weeks 1–14
- **Class notes (10%)**: 50–100 word summary within 1 day. Each note's cheat sheet has a template.

## Running the code
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk jupyter
cd code
python make_figures.py                 # regenerates images/
jupyter notebook                       # open the .ipynb files
```
