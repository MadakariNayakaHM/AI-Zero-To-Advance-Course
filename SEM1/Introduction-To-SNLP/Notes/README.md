# Introduction to Speech & Natural Language Processing

> Dr. Krishnendu Ghosh · IIIT Dharwad · Semester 1 · 1-credit elective

## Notes

| # | Topic | Lecture | Notebook |
|:-:|:--|:--|:--|
| 01 | [What is SNLP? The Information Layers of Language](01-Information-Layers-of-SNLP.md) | Lectures 0–1 | [deep dive](code/01_language_layers_deep_dive.ipynb) |
| 02 | [Lexical Processing, Part 1](02-Lexical-Processing-Part-1.md) | Lecture 2 · 30 Sep | [deep dive](code/02_lexical_deep_dive.ipynb) |
| 03 | [Lexical Processing, Part 2: Minimum Edit Distance](03-Minimum-Edit-Distance.md) | Lecture 3 | [deep dive](code/03_edit_distance_deep_dive.ipynb) |
| 04 | [Syntactic Processing: Constituency & Dependency Parsing](04-Syntactic-Processing-Parsing.md) | Lecture 4 | [deep dive](code/04_parsing_deep_dive.ipynb) |

Each note has worked examples, a bank of practice problems with full solutions, and a cheat sheet. Notes 01–02 also have real-world case studies; Notes 03–04 walk through every slide example step by step (DP tables, the CKY chart, the arc-eager trace).

**Class examples notebook:** [lexical processing, hands-on](code/lexical_processing_hands_on.ipynb) covers types & tokens, Heaps' law, stemming vs lemmatization, a toy morphological parser, sentence segmentation, tokenizer comparison, BPE from scratch, an ambiguous parse tree and word senses.

**Coming up:** n-grams · BoW, TF-IDF, Word2Vec, GloVe · speech processing · ASR & TTS · POS & NER · LLMs

## Assessment

| Component | Weight | Date |
|:--|--:|:--|
| Quizzes (12) | 36% | after each lecture |
| Theoretical assignment 1 | 12% | 14 Oct |
| Theoretical assignment 2 | 12% | 9 Dec |
| Project formulation | 20% | presentation 23 Dec |
| Attendance | 20% | — |

## Worth knowing about the slide examples

- *"The cat chased the cat"* has 3 types only after lower-casing; case-sensitive, it has 4.
- WordNet lemmatizes *better* → *good*, but keeps *best* as *best*.
- The single Porter rule ATIONAL → ATE gives *relate*; the full Porter stemmer then drops the final *e* (step 5a), giving *relat*.
- The CKY chart writes `V → tanks 0.1`, but the grammar (and the chart's own VP → V 0.03) use **0.3**.
- *people fish with rods* only parses because of the rule `NP → e` (an empty object NP).
- On the *Adding Backtrace* slide, LEFT and DOWN are swapped: coming from the left is an **insertion**, from below a **deletion**.
- The slides call the substitution-cost-2 version "Levenshtein"; most libraries use cost 1 under that name, so say which costs you use.

## Textbooks

Jurafsky & Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) (3rd ed., free) · Bird, Klein & Loper, [*Natural Language Processing with Python*](https://www.nltk.org/book/) · Benesty et al., *Springer Handbook of Speech Processing*
