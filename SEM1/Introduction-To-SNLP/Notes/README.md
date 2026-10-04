# Introduction to Speech & Natural Language Processing: Notes

**Instructor:** Dr. Krishnendu Ghosh · IIIT Dharwad · 1-credit elective (1-0-0-0-1), Wednesdays
**Style of course:** high-level: applications, pipelines, intuition; small toy models; not deep theory.

## Key dates & grading

| Component | Weight | Date |
|---|---|---|
| Theoretical Assignment 1 | 12% (of 24%) | **14 Oct 2026** (based on Lecture 2) |
| Theoretical Assignment 2 | 12% | 9 Dec 2026 |
| **Quizzes (12)** | **36%** | One per lecture; ~1 week to submit |
| Project formulation | 20% | Presentation **23 Dec 2026** |
| Attendance | 20% / 10% | Semester ends 27 Dec |

## Lecture tracker

| # | Note | Lecture | Sources | Status |
|---|---|---|---|---|
| 01 | [What is SNLP? History & Information Layers](01-Information-Layers-of-SNLP.md) | L0 (course intro) + L1 | PPT | ✅ (no transcript) |
| 02 | [Lexical Processing, Part 1](02-Lexical-Processing-Part-1.md) | L2, 30 Sep | PPT + transcript (both parts) | ✅ (§9–11 continue next class) |
| 03 | Lexical Processing, Part 2 (segmentation, tokenization, normalization, n-grams?) | next | — | ⏳ |

Syllabus: Unit 1 What is SNLP · Unit 2 Basic speech processing · Unit 3 Basic text processing (tokenization → TF-IDF, Word2Vec, GloVe) · Unit 4 Speech applications (ASR, Whisper/wav2vec, TTS) · Unit 6 NLP applications (classification, POS, NER, parsing, LLMs). *The slides skip "Unit 5".*

**Notebook:** [`code/lexical_processing_hands_on.ipynb`](code/lexical_processing_hands_on.ipynb) covers types/tokens & TTR, Heaps' law, Porter vs WordNet lemmatizer, a toy morphological parser, sentence segmentation, tokenizer comparison, case folding, **BPE from scratch**, a constituency parse with ambiguity, and WordNet senses of "bank".
**Professor's Colab (Lecture 2):** [link](https://colab.research.google.com/drive/1I5S7q_jiuAACft0xSXBos0qJwvWnYGKn?usp=sharing)

## Things to watch for in the slides

| Where | Note |
|---|---|
| L2 "The cat chased the cat" = 3 types | True only after **lower-casing**; case-sensitive = 4 |
| L2 "Better, best → lemma good" | WordNet gives good for *better*, but keeps *best* as *best* |
| L2 Porter "relational → relate" | The single rule ATIONAL→ATE gives *relate*; the full Porter cascade gives *relat* |

## Cross-course links

| Topic | Also in |
|---|---|
| NLTK tokenization & POS tagging in an ML pipeline | [MLP Note 02](../../Machine-Learning-Paradigms/Notes/02-ML-Pipeline-Hands-On.md) |
| Tokens / softmax over vocabulary in LLMs | [Gen AI Note 02](../../Introduction-To-Generative-AI/Notes/02-Neural-Networks-Fundamentals.md) |

## Running the code
```bash
pip install nltk matplotlib numpy jupyter
cd code && jupyter notebook lexical_processing_hands_on.ipynb   # downloads NLTK data on first run
```
