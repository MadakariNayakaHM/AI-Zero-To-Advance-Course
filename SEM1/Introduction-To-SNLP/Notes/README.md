# Introduction to Speech & Natural Language Processing

> Dr. Krishnendu Ghosh · IIIT Dharwad · Semester 1 · 1-credit elective

## Notes

| # | Topic | Lecture |
|:-:|:--|:--|
| 01 | [What is SNLP? The Information Layers of Language](01-Information-Layers-of-SNLP.md) | Lectures 0–1 |
| 02 | [Lexical Processing, Part 1](02-Lexical-Processing-Part-1.md) | Lecture 2 · 30 Sep |

**Notebook:** [lexical processing, hands-on](code/lexical_processing_hands_on.ipynb) covers types & tokens, Heaps' law, stemming vs lemmatization, a toy morphological parser, sentence segmentation, tokenizer comparison, BPE from scratch, an ambiguous parse tree and word senses.

**Coming up:** tokenization & normalization (part 2) · n-grams · BoW, TF-IDF, Word2Vec, GloVe · speech processing · ASR & TTS · POS, NER & parsing · LLMs

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
- The single Porter rule ATIONAL → ATE gives *relate*; the full Porter stemmer gives *relat*.

## Textbooks

Jurafsky & Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) (3rd ed., free) · Bird, Klein & Loper, [*Natural Language Processing with Python*](https://www.nltk.org/book/) · Benesty et al., *Springer Handbook of Speech Processing*
