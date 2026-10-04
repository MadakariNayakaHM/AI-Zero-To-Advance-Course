# 02 · Assignments Guide & Practical Research Toolkit

> **Course:** Introduction to Research (I2R) · IIIT Dharwad
>
> **Sources:** assignment slides 36–38 of `PPT-1.pdf` and the 29 Sep discussion. Everything else is a **practical toolkit (beyond slides)** to help you do the assignments and your M.Tech project well.
>
> ⚠️ **This note gives structure, methods and templates, not answers.** The assignments are graded individual work, and the course has an ethics & plagiarism component. Use these as scaffolding and write your own content.

---

## 📌 Table of Contents

1. [The Three Assignments at a Glance](#1-the-three-assignments-at-a-glance)
2. [Assignment 1: Planning Your Research (Phases I–IV)](#2-assignment-1-planning-your-research-phases-iiv)
3. [Assignment 2: GenAI Tools for Research](#3-assignment-2-genai-tools-for-research)
4. [Assignment 3: Topic Exploration & a 5-Paper Literature Review](#4-assignment-3-topic-exploration--a-5-paper-literature-review)
5. [Templates](#5-templates)
6. [From Gap to Research Question to Hypothesis](#6-from-gap-to-research-question-to-hypothesis)
7. [ML/DS-Specific Research Hygiene](#7-mlds-specific-research-hygiene)
8. [Tool Directory (verified links)](#8-tool-directory-verified-links)

---

## 1. The Three Assignments at a Glance

| # | Topic | Time | Weight | Core deliverable |
|---|---|---|---|---|
| **A1** | **How to plan your research work**: focus of Phase I, II, III, IV | 4 weeks | 10% | A phased plan for your M.Tech research |
| **A2** | **How to use GenAI tools for research**: literature review, research exploration, slide decks, reports | 8 weeks | 10% | Tool evaluation / case study across 4 stages |
| **A3** | **Topic exploration & literature review** (prototype study): pick a problem (with a supervisor or your own), find **5 papers**, write a **literature review report** | 10 weeks | 10% | A literature review report on 5 papers |

> 💡 A1 → A3 → your semester-1 project evaluation all point the same way: **problem statement + literature review by the end of semester 1**. Do the assignments **on your actual M.Tech topic**: one effort, three payoffs.

---

## 2. Assignment 1: Planning Your Research (Phases I–IV)

The slide asks for the **focus of Phases I–IV**. A natural mapping onto the 4-semester M.Tech, consistent with the lecture:

| Phase | Semester | Credits / hours | Focus (from lecture) | Typical outputs |
|---|---|---|---|---|
| **I** | 1 | 3 cr · ~75 h | Broad area → sub-field → **literature review & critical review** → list questions → **finalise the problem statement** | Problem statement, literature review, (maybe a review paper with your mentor) |
| **II** | 2 | 6 cr · ~150 h | Hypothesise solutions; explore; **first novel result** | Conference paper / arXiv preprint |
| **III** | 3 | 9 cr · ~225 h | **Second novel result**; extend toward a **journal-quality** study; **one research article** | Journal / conference paper, poster at the Research Conclave |
| **IV** | 4 | 12 cr · ~300 h | **Third novel result**, consolidation, **thesis writing & defence**, future scope | Thesis, defence, final paper/patent |

(The lecture's *supervisor phases*, I defined → II fuzzy → III independent, run alongside: you should become more independent each semester.)

**What a strong plan includes (checklist):**

- [ ] Broad area + why it matters (link to a thrust area: health, energy, education, agriculture…)
- [ ] Narrowed sub-field and a **candidate problem statement** (one sentence)
- [ ] **Feasibility**: data access, compute, tools, mentor expertise (remember the EEG example!)
- [ ] Courses for semesters 2–4 that support the topic
- [ ] Per-phase **objectives, activities, deliverables, milestones** (a Gantt-style table)
- [ ] **Risks & fallback plans** (e.g. "if hospital data is unavailable → use the public dataset X")
- [ ] Target venues (conference first, then journal; check CORE / SCImago)
- [ ] Mentor meeting cadence (weekly) and research-diary habit
- [ ] Ethics: data privacy, consent, plagiarism, responsible GenAI use

---

## 3. Assignment 2: GenAI Tools for Research

The slide lists **four stages**. For each, evaluate tools on **what they do well, failure modes, and responsible use**.

| Stage | Tool categories | Example tools (verify each yourself) |
|---|---|---|
| **Literature review** | Semantic search, citation graphs, paper Q&A, summarisation | Semantic Scholar, Connected Papers, Research Rabbit, Elicit, Consensus, NotebookLM, ChatGPT/Claude with uploaded PDFs |
| **Research exploration** | Brainstorming, code assistants, experiment design, data analysis | Claude/ChatGPT/Gemini, GitHub Copilot / Claude Code, Jupyter AI, Hugging Face Papers (papers + code) |
| **Slide-deck preparation** | Outline → slides, figure generation | Gamma, Copilot in PowerPoint, Canva Magic, Beamer + LLM help |
| **Report preparation** | Drafting, editing, LaTeX, references | Overleaf (+ AI assist), Grammarly, Zotero / Mendeley for references, LLMs for language polishing |

**A strong A2 report goes beyond listing tools:**

1. Run the **same task** (e.g. "find key papers on X") on 2–3 tools and **compare** coverage, accuracy and speed.
2. **Document failure cases.** The #1 risk: **hallucinated citations** (papers that don't exist, wrong authors/years). Always verify the DOI/arXiv ID.
3. **Ethics & policy:** disclose AI use; never submit AI text as your own original contribution; don't upload confidential, employer or patient data to public tools; check venue policies on AI-written text (many forbid listing AI as an author).
4. Reflect: where did AI **save time** and where did it **mislead**? (This connects to the course's ethics unit and the reference book *Generative AI for PhD Scholars: Ethical Use of AI in Research*.)

> 🔗 Your Gen AI course notes explain *why* LLMs hallucinate: they sample likely next tokens and don't look facts up (Gen AI Note 01 §11).

---

## 4. Assignment 3: Topic Exploration & a 5-Paper Literature Review

### Step 1: topic exploration
Broad area → sub-field → specific problem. Use survey papers, recent top-venue proceedings and PhD theses (Shodhganga) to map the landscape.

### Step 2: search strategy (be systematic)

- Build a **keyword set** with synonyms: e.g. ("speech emotion recognition" OR "SER") AND ("low-resource" OR "Indian languages") AND ("self-supervised" OR "wav2vec").
- Search **Google Scholar, Semantic Scholar, IEEE Xplore, ACM DL, arXiv**.
- **Snowballing:** follow references **backward** (what they cite) and citations **forward** (who cites them). Connected Papers / Research Rabbit visualise this.
- Prefer **recent (≈ last 3–5 years) + seminal** papers from **good venues** (Q1/Q2 journals, CORE A*/A conferences).

### Step 3: select 5 papers (a balanced mix)

| Slot | Paper type |
|---|---|
| 1 | A **survey / review** (the map of the field) |
| 2 | The **seminal / most-cited** method |
| 3–4 | **Recent state-of-the-art** methods (different approaches) |
| 5 | A paper closest to **your intended problem / setting** (data, language, domain) |

### Step 4: read efficiently with Keshav's 3-pass method

| Pass | Time | Read | Goal |
|---|---|---|---|
| 1 | 5–10 min | Title, abstract, intro, section headings, conclusion | **The 5 Cs**: Category, Context, Correctness, Contributions, Clarity → keep or drop? |
| 2 | ~1 h | Figures, tables, main arguments (skip proofs) | Summarise the method and evidence |
| 3 | 4–5 h | Everything; mentally re-implement | Find hidden assumptions, weaknesses, **gaps** |

### Step 5: the literature matrix (the secret to a good review)
Fill one row per paper (template in §5.3), then write the review **by theme, not paper-by-paper**:

- ❌ "Paper 1 did… Paper 2 did… Paper 3 did…" (annotated bibliography)
- ✅ "Two families of approaches exist: A (papers 1, 3) and B (2, 4). A achieves… but fails when…; B… However, none of them address **[gap]**."

### Step 6: suggested report structure

1. Introduction: the problem and why it matters
2. Search methodology: databases, keywords, inclusion criteria
3. Background / key concepts
4. **Thematic review** of the 5 papers (with the comparison table)
5. **Critical analysis**: strengths, limitations (be *positively* critical!)
6. **Research gaps** → candidate research questions
7. Conclusion & future directions
8. References (consistent style, e.g. IEEE; managed with Zotero)

> ⚠️ **Plagiarism:** paraphrase in your own words and **cite every idea** you took (syllabus: *paraphrasing and giving credit to the original authors*). AI-paraphrasing someone's text without citing them is still plagiarism.

---

## 5. Templates

### 5.1 Weekly mentor-meeting agenda (5 minutes to prepare)
```markdown
## Meeting — <date> — <mentor>
**Since last meeting (recap):** decisions/actions agreed on <prev date>
**Progress:** 1. … 2. …  (attach 1 plot/table if any)
**Blockers / questions:** 1. … 2. …
**Proposed next steps:** …
**Asks from mentor:** feedback on X / pointer to Y
```

### 5.2 Research diary entry
```markdown
### <date> — <topic>
- Goal today:
- What I did / read:
- Result / observation (incl. surprises & failures):
- Why I think it happened:
- Decision / next step:
- Open questions for mentor:
- Links (code commit, notebook, paper):
```

### 5.3 Literature matrix (for A3 and your project)

| # | Citation (venue, year, tier) | Problem addressed | Method / key idea | Data | Metrics & main result | Strengths | Limitations / assumptions | Gap it leaves | Relevance to me |
|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |

### 5.4 One-line problem statement formula
> *"Existing **[approach]** for **[task]** on **[data/setting]** suffers from **[limitation]**. We investigate whether **[idea]** can **[measurable improvement]**, evaluated by **[metric]** on **[dataset]**."*

---

## 6. From Gap to Research Question to Hypothesis

The syllabus moves **gaps → questions → hypotheses → exploration**. Make each one concrete:

| Stage | Weak | Strong |
|---|---|---|
| **Gap** | "Not much work on Kannada ASR" | "Self-supervised ASR models reach <10% WER for English but >30% WER for Kannada with < 50 h labelled data (papers 2, 4)" |
| **Research question** | "Can we improve Kannada ASR?" | "Does continued pre-training of wav2vec 2.0 on 500 h of *unlabelled* Kannada audio reduce WER in the 10 h-labelled setting?" |
| **Hypothesis** (testable, falsifiable) | "It will be better" | "Continued pre-training reduces WER by ≥ 15% relative vs the multilingual baseline, at equal fine-tuning data" |
| **Exploration** | "Train a model" | Baseline vs proposed, 3 random seeds, ablation on pre-training hours (100/250/500 h), significance test |

**A good RQ is FINER:** Feasible, Interesting, Novel, Ethical, Relevant.

---

## 7. ML/DS-Specific Research Hygiene

Things reviewers check in DS/AI papers (and your M.Tech committee will too):

| Practice | Why |
|---|---|
| **Strong baselines** (incl. simple ones) | Shows your improvement isn't trivial (cf. the "predict the mean" baseline in MLP Note 03) |
| **Proper train/val/test split**; no test-set tuning; no leakage | Honest generalisation (MLP Note 02) |
| **Multiple random seeds**, mean ± std | Results vary with the seed (MLP Note 02 §12) |
| **Ablation studies** | Show *which* component causes the gain |
| **Statistical significance** | A 0.3% gain may be noise |
| **Reproducibility**: code, configs, data versions, `requirements.txt` | Others (and future you) can re-run it |
| **Error analysis** | Where and why it fails → the next research question |
| **Report negative results honestly** | *"If you can explain why it didn't work, that's research too."* |

---

## 8. Tool Directory (verified links)

| Purpose | Tool |
|---|---|
| Search | [Google Scholar](https://scholar.google.com/) · [Semantic Scholar](https://www.semanticscholar.org/) · [arXiv](https://arxiv.org/) |
| Citation maps | [Connected Papers](https://www.connectedpapers.com/) · [Research Rabbit](https://www.researchrabbit.ai/) |
| AI literature assistants | [Elicit](https://elicit.com/) · [Consensus](https://consensus.app/) (always verify citations) |
| Papers with code | [Hugging Face Papers](https://huggingface.co/papers) |
| Reference management | [Zotero](https://www.zotero.org/) (free, browser plug-in, BibTeX export) |
| Writing (LaTeX) | [Overleaf](https://www.overleaf.com/) |
| Venue quality | [CORE conference ranks](https://portal.core.edu.au/conf-ranks/) · [SCImago journal quartiles](https://www.scimagojr.com/) |
| Indian theses | [Shodhganga](https://shodhganga.inflibnet.ac.in/) |
| Patents (India) | [IP India](https://ipindia.gov.in/) |
| Reading method | [Keshav — How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) |
| Mindset | [Hamming — You and Your Research](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html) · [Bellare — The Ph.D Experience](https://cseweb.ucsd.edu/~mihir/phd.html) |

---
⬅️ [01 · Why Research & the Research Path](01-Why-Research-and-the-Research-Path.md) · [Research Index](README.md)
