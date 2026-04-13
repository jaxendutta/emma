# EMMA: Emergency Medicine Mentoring Agent

## Group 23

| Member              | Email                 | ID        |
| ------------------- | --------------------- | --------- |
| Jaxen Anirban Dutta | <adutt042@uottawa.ca> | 300101437 |
| Acassia Arnaud      | <aarna035@uottawa.ca> | 300466030 |
| Yifei Yu            | <yyu039@uottawa.ca>   | 8719434   |

## Overview

EMMA is a conversational medical study agent for USMLE preparation. In explanation mode, students pose clinical questions in natural language and receive responses grounded in passages from 18 standard medical textbooks. In quiz mode, EMMA presents authentic USMLE-style questions, evaluates answers, and tracks per-specialty performance. A collaborative filtering recommender steers students toward their weakest areas.

**Live client:** [emma.vercel.app](https://emma.vercel.app) *(static client + Dialogflow chatbot)*

## Contents

- [EMMA: Emergency Medicine Mentoring Agent](#emma-emergency-medicine-mentoring-agent)
  - [Group 23](#group-23)
  - [Overview](#overview)
  - [Contents](#contents)
  - [1. Architecture Pipeline](#1-architecture-pipeline)
  - [2. Repository Structure](#2-repository-structure)
  - [3. Notebooks](#3-notebooks)
  - [4. Data](#4-data)
  - [5. Vectorstore](#5-vectorstore)
    - [5.1. Relevant Files](#51-relevant-files)
    - [5.2. Getting `vectorstore` Files](#52-getting-vectorstore-files)
    - [5.3. Retrieval Quality](#53-retrieval-quality)
  - [6. NER \& Query Rewriting](#6-ner--query-rewriting)
    - [6.1. Relevant Files](#61-relevant-files)
    - [6.2. NER Model](#62-ner-model)
      - [6.2.1. Model](#621-model)
      - [6.2.2. Labels](#622-labels)
      - [6.2.3. Install](#623-install)
      - [6.2.4. Corpus Statistics (MedQA train, 10,178 questions)](#624-corpus-statistics-medqa-train-10178-questions)
      - [6.2.5. NER Rewriting Impact on FAISS Retrieval Score](#625-ner-rewriting-impact-on-faiss-retrieval-score)
  - [7. Classification](#7-classification)
    - [7.1. Relevant Files](#71-relevant-files)
    - [7.2. Task](#72-task)
    - [7.3. Champion: TF-IDF Bigrams + LinearSVC](#73-champion-tf-idf-bigrams--linearsvc)
  - [8. Clustering](#8-clustering)
    - [8.1. Relevant Files](#81-relevant-files)
    - [8.2. Method](#82-method)
    - [8.3. Interpretation of Near-Zero $\\kappa$](#83-interpretation-of-near-zero-kappa)
  - [9. Recommender System](#9-recommender-system)
    - [9.1. Relevant Files](#91-relevant-files)
    - [9.2. Task](#92-task)
    - [9.3. Algorithms Evaluated](#93-algorithms-evaluated)
    - [9.4. Evaluation](#94-evaluation)
    - [9.5. Champion: KNNBasic](#95-champion-knnbasic)
  - [10. RAG Pipeline \& Benchmarks](#10-rag-pipeline--benchmarks)
    - [10.1. Relevant Files](#101-relevant-files)
    - [10.2. Benchmark combinations](#102-benchmark-combinations)
    - [10.3. Finding](#103-finding)
  - [11. FastAPI Backend](#11-fastapi-backend)
    - [11.1. Relevant Files](#111-relevant-files)
    - [11.2. Endpoints](#112-endpoints)
    - [11.3. Two-Turn Async Pattern](#113-two-turn-async-pattern)
  - [12. Setup](#12-setup)
    - [12.1. Prerequisites](#121-prerequisites)
    - [12.2. Install](#122-install)
    - [12.3. Environment variables](#123-environment-variables)
    - [12.4. Pull the LLM (Ollama)](#124-pull-the-llm-ollama)
    - [12.5. Open notebooks](#125-open-notebooks)
  - [13. Running the API Locally](#13-running-the-api-locally)
    - [13.1. Start the server](#131-start-the-server)
    - [13.2. Expose to Dialogflow via ngrok](#132-expose-to-dialogflow-via-ngrok)
    - [13.3. Test Directly (No DialogFlow)](#133-test-directly-no-dialogflow)
  - [14. Key Design Decisions](#14-key-design-decisions)
  - [References](#references)

## 1. Architecture Pipeline

```mermaid
flowchart TD
    U(["User Query"])

    subgraph NER ["NB5: NER + Query Rewriting"]
        direction TB
        N1["SpaCy en_ner_bc5cdr_md
        extract DISEASE + CHEMICAL entities"]
        N2{"Entities
        found?"}
        N3["Rewritten query
        = entity string"]
        N4["Raw query
        fallback: 5.8% of questions"]
        N1 --> N2
        N2 -->|yes| N3
        N2 -->|no| N4
    end

    subgraph VS ["NB01: Vectorstore"]
        direction TB
        TB[(18 Medical Textbooks
        36,723 chunks · 1024-dim)]
        V1["FAISS IndexFlatIP
        Octen-Embedding-0.6B"]
        V2{"Score band?"}
        V3["high ≥ 0.70"]
        V4["medium 0.55–0.70
        flagged in prompt"]
        V5["low < 0.55
        dropped"]
        TB -.->|"Octen-Embedding-0.6B
        at build time"| V1
        V1 --> V2
        V2 --> V3
        V2 --> V4
        V2 --> V5
    end

    subgraph CLS ["NB02: Classifier"]
        C1["TF-IDF Bigrams + LinearSVC
        19 specialty labels
        F1 = 0.69 · κ = 0.66"]
    end
  
    subgraph CLU ["NB03: Clustering"]
        K1["BERTopic
        39 fine-grained topics
        C_v = 0.475"]
    end

    subgraph LLM ["LLM Inference"]
        direction TB
        MC[(config/models.json
        benchmark_combinations)]
        L4["Structured prompt:
        - retrieved passages
        - specialty context
        - confidence hedging"]
        L1{"Ollama
        running?"}
        L2["Ollama
        qwen3:4b-thinking-2507
        fast · no GPU needed"]
        L3["HuggingFace
        Qwen3-4B-Thinking-2507
        4-bit nf4 · Colab T4"]
        MC -.->|model selection| L1
        L4 --> L1
        L1 -->|yes| L2
        L1 -->|no| L3
    end

    subgraph CRS ["NB6: Recommender System"]
        R1["KNNBasic CF
        per-specialty accuracy tracking
        HR@10 = 0.740"]
    end

    A(["Answer grounded
    in textbook passages"])

    U --> N1
    N3 --> V1
    N4 --> V1
    U --> C1
    C1 --> CLU
    V3 --> L4
    V4 --> L4
    C1 --> L4
    CLU --> L4
    L2 --> A
    L3 --> A
    A --> CRS
```

Clinical vignettes score lower in raw FAISS retrieval because incidental language ("A 45-year-old man presents with...") dilutes the embedding. NER rewriting isolates the DISEASE and CHEMICAL tokens before querying, improving retrieval scores by +0.005–0.006 on biomedical embeddings.

## 2. Repository Structure

```plain
emma/
├── config/
│   └── models.json              # single source of truth: LLMs, embeddings, benchmark grid
├── data/
│   ├── MedQA-USMLE/
│   │   ├── questions/           # train/dev/test JSONL (10,178 / 1,273 / 1,273 questions)
│   │   └── textbooks/en/        # 18 medical textbooks (.txt)
│   └── MedMCQA/                 # train/validation/test parquet (182k questions)
├── models/
│   ├── vectorstore/             # FAISS index per embedding model (gitignored, ~143 MB each)
│   ├── classifier/              # tfidf_svm.pkl, label_encoder.pkl
│   ├── ner/                     # entity_stats.json, collocations, config.json
│   ├── clustering/              # BERTopic model
│   ├── recommender/             # ratings.csv, results.json
│   ├── rag/                     # per-run results.parquet + config.json
│   └── benchmarks.json          # ablation grid results (committed to git)
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_vectorstore_build.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_rag_pipeline.ipynb
│   ├── 05_ner.ipynb
│   ├── 06_crs.ipynb
│   └── 07_evaluation_benchmark.ipynb
├── src/
│   ├── data.py                  # data loaders (MedQA, MedMCQA, textbooks)
│   ├── vectorstore.py           # FAISS build + search
│   ├── retrieval.py             # EMMARetriever: NER -> FAISS -> classify -> LLM
│   ├── classify.py              # classification pipeline
│   ├── cluster.py               # BERTopic evaluation
│   └── api.py                   # FastAPI webhook
├── client/                      # static web app (deployed on Vercel)
├── run_api.py                   # API server entrypoint
├── pyproject.toml
└── scripts/
    ├── setup.sh                 # Unix / WSL setup
    └── setup.ps1                # Windows PowerShell setup
```

## 3. Notebooks

| #   | Notebook                        | Purpose                                                                  | Runs on        |
| --- | ------------------------------- | ------------------------------------------------------------------------ | -------------- |
| 0   | `00_data_exploration.ipynb`     | Dataset EDA: textbook sizes, MedQA/MedMCQA distributions                 | Local          |
| 1   | `01_vectorstore_build.ipynb`    | Chunk textbooks > embed > build FAISS index                              | Colab T4       |
| 2   | `02_classification.ipynb`       | Feature × classifier grid on MedMCQA, champion selection                 | Local or Colab |
| 3   | `03_clustering.ipynb`           | BERTopic + GMM + Spectral on MedQA questions                             | Local or Colab |
| 4   | `04_rag_pipeline.ipynb`         | End-to-end RAG pilot: NER → FAISS → LLM (50 questions)                   | **Colab T4**   |
| 5   | `05_ner.ipynb`                  | NER corpus analysis, collocation, retrieval score comparison             | Local          |
| 6   | `06_crs.ipynb`                  | Collaborative filtering recommender (SVD, NMF, KNNBasic)                 | Local          |
| 7   | `07_evaluation_benchmark.ipynb` | Full ablation grid: 6 combinations of embeddings × LLMs × RAG conditions | Colab T4       |

All notebooks auto-detect Colab and load artefacts from Google Drive. They resume from checkpoint if the session is interrupted.

## 4. Data

Three sources, all committed to `data/`:

| #   | Dataset                                        | Questions             | Purpose                                    |
| --- | ---------------------------------------------- | --------------------- | ------------------------------------------ |
| 1   | [MedQA-USMLE](https://github.com/jind11/MedQA) | 12,723 (train 10,178) | RAG evaluation, clustering, NER analysis   |
| 2   | [MedMCQA](https://github.com/MedMCQA/MedMCQA)  | 179,777               | Classifier training (has specialty labels) |
| 3   | 18 medical textbooks                           | 36,723 chunks         | RAG retrieval corpus                       |

MedMCQA is used for classifier training only. Its `subject_name` labels provide the specialty ground truth that MedQA lacks. The textbooks were written by the same authors who wrote the MedQA questions, making them the ideal retrieval source.

```python
from src.data import load_medqa, load_medmcqa, load_all_textbooks
df    = load_medqa(split='train')   # 10,178 rows
books = load_all_textbooks()        # dict of 18 textbooks
```

## 5. Vectorstore

### 5.1. Relevant Files

| #   | File                                   | Purpose                 |
| --- | -------------------------------------- | ----------------------- |
| 1   | `src/vectorstore.py`                   | build + query functions |
| 2   | `notebooks/01_vectorstore_build.ipynb` | run once on Colab T4    |

```mermaid
graph LR
    A[18 Textbooks] --> B[Chunking: 400 words, 50 overlap]
    B --> C[Embedding Model]
    C --> D[FAISS IndexFlatIP]
    D --> E[models/vectorstore/embedding_id/]
```

Three vectorstores were built and evaluated (one per embedding model):

| #   | Embedding            | Dim  | RTEB Healthcare rank | Default                   |
| --- | -------------------- | ---- | -------------------- | ------------------------- |
| 1   | Octen-Embedding-0.6B | 1024 | #15                  | No (best ablation result) |
| 2   | Qwen3-Embedding-0.6B | 1024 | #177                 | Yes (build default)       |
| 3   | all-MiniLM-L12-v2    | 384  | —                    | No                        |

### 5.2. Getting `vectorstore` Files

The index files are too large for git (~143 MB each). Three options:

1. **Download pre-built** — use the auto-download cell in NB01 Section 4 (pulls from shared Google Drive)
2. **Rebuild on Colab** — run NB01 on a T4 GPU (~45 min per embedding model)
3. **Local rebuild** — run NB01 locally if you have a GPU with ≥8GB VRAM

Place files under `models/vectorstore/<embedding_id>/`:

```plain
models/vectorstore/
  octen-embedding-0.6b/
    index.faiss
    texts.pkl
    metadata.pkl
    config.json
```

### 5.3. Retrieval Quality

| #   | Query type                                         | Score range | Confidence band |
| --- | -------------------------------------------------- | ----------- | --------------- |
| 1   | Direct question (e.g. "anaphylaxis mechanism")     | 0.72–0.73   | high            |
| 2   | Direct question (e.g. "beta blocker side effects") | 0.72–0.73   | high            |
| 3   | Raw clinical vignette                              | 0.63–0.66   | medium          |
| 4   | NER-rewritten vignette (Octen)                     | 0.65–0.66   | medium          |

## 6. NER & Query Rewriting

### 6.1. Relevant Files

| #   | File                     | Purpose                                                                                                  |
| --- | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| 1   | `src/retrieval.py`       | `NER_MODEL`, `ENTITY_LABELS`, `extract_entities()`, `rewrite_query()`, NER and query rewriting functions |
| 2   | `notebooks/05_ner.ipynb` | NER corpus analysis and retrieval score validation                                                       |

### 6.2. NER Model

#### 6.2.1. Model

`en_ner_bc5cdr_md` (BC5CDR corpus, 1,500 PubMed articles)

#### 6.2.2. Labels

- `DISEASE`
- `CHEMICAL`

> [!NOTE]
> **Why not `en_core_sci_md`?**
> That model outputs a single generic `ENTITY` label. It cannot distinguish between diseases and chemicals. `en_ner_bc5cdr_md` is the only ScispaCy model that produces typed biomedical entities suitable for query rewriting.

#### 6.2.3. Install

```bash
# Already in pyproject.toml, installed by uv sync
# To install manually:
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

#### 6.2.4. Corpus Statistics (MedQA train, 10,178 questions)

- 54,256 total entities extracted
- DISEASE: 39,575 | CHEMICAL: 14,681
- Mean 5.33 entities per question
- 593 questions (5.8%) have zero entities → fall back to raw query

#### 6.2.5. NER Rewriting Impact on FAISS Retrieval Score

| #   | Embedding            | Raw vignette | NER rewrite | Delta  |
| --- | -------------------- | ------------ | ----------- | ------ |
| 1   | all-MiniLM-L12-v2    | 0.5412       | 0.5191      | -0.022 |
| 2   | Qwen3-Embedding-0.6B | 0.6379       | 0.6431      | +0.005 |
| 3   | Octen-Embedding-0.6B | 0.6525       | 0.6584      | +0.006 |

NER rewriting helps biomedical-scale embeddings and hurts general-purpose ones. Model and NER strategy must be co-designed.

## 7. Classification

### 7.1. Relevant Files

| #   | File                                  | Purpose                                   |
| --- | ------------------------------------- | ----------------------------------------- |
| 1   | `src/classify.py`                     | feature pipelines, CV, training           |
| 2   | `notebooks/02_classification.ipynb`   | full feature × classifier grid            |
| 3   | `models/classifier/tfidf_svm.pkl`     | fitted champion pipeline (TF-IDF + SVM)   |
| 4   | `models/classifier/label_encoder.pkl` | fitted LabelEncoder for specialty classes |

### 7.2. Task

19-class specialty prediction on MedMCQA questions. Used to route each query to the correct specialty context at inference time.

### 7.3. Champion: TF-IDF Bigrams + LinearSVC

| #   | Metric      | 10-fold CV (20k sample) | Holdout (full 179k) |
| --- | ----------- | ----------------------- | ------------------- |
| 1   | Weighted F1 | 0.5424 ± 0.0086         | 0.69                |
| 2   | Cohen's κ   | 0.5089 ± 0.0096         | 0.66                |

Mean inter-category cosine similarity: 0.72 (vs. ~0.95 in the A1 corpus), confirming the task is tractable for a linear classifier.

## 8. Clustering

### 8.1. Relevant Files

| #   | File                            | Purpose                        |
| --- | ------------------------------- | ------------------------------ |
| 1   | `src/cluster.py`                | BERTopic evaluation helpers    |
| 2   | `notebooks/03_clustering.ipynb` | clustering analysis and models |

### 8.2. Method

BERTopic (MiniLM-L6-v2 embeddings → UMAP → HDBSCAN). Auto-discovers K=39 topics.

| #   | Method                | Cohen's $\kappa$ | Silhouette | $C_v$ coherence |
| --- | --------------------- | ---------------: | ---------: | --------------: |
| 1   | TF-IDF + GMM          |            0.014 |          — |               — |
| 2   | Embeddings + Spectral |            0.024 |      0.064 |               — |
| 3   | BERTopic              |           -0.020 |      0.072 |           0.475 |

### 8.3. Interpretation of Near-Zero $\kappa$

BERTopic discovers 39 fine-grained topic groups that do not align one-to-one with 19 specialty labels. This is granularity mismatch, not failure. C_v = 0.475 confirms internal topic coherence. Topic 0 (chest/cardiac terms) is 70.6% Internal Medicine; Topic 4 (gestation/pregnancy) is 72.4% Obstetrics. The 36% outlier rate reflects short question stems (~20 words) that don't form dense HDBSCAN clusters — these fall back to specialty-only routing.

## 9. Recommender System

### 9.1. Relevant Files

| #   | File                     | Purpose                        |
| --- | ------------------------ | ------------------------------ |
| 1   | `notebooks/06_crs.ipynb` | Recommender system development |
| 2   | `models/recommender/`    | ratings, results, and config   |

### 9.2. Task

Recommend which specialties a student should focus on, based on their quiz history. Collaborative filtering finds latent weakness patterns across students.

### 9.3. Algorithms Evaluated

| #   | Algorithm       | Type                            |
| --- | --------------- | ------------------------------- |
| 1   | SVD             | Matrix factorization            |
| 2   | NMF             | Matrix factorization            |
| 3   | KNNBasic        | Memory-based                    |
| 4   | NormalPredictor | Baseline (predicts mean rating) |

### 9.4. Evaluation

Used a synthetic dataset of 200 students with randomized quiz histories. Evaluated on RMSE and Hit Rate @ K (whether the model's top K recommendations include at least one of the student's actual weak specialties).

### 9.5. Champion: KNNBasic

| #   | Metric           | KNNBasic | NormalPredictor |
| --- | ---------------- | -------: | --------------: |
| 1   | RMSE (5-fold CV) |   0.2208 |          0.3109 |
| 2   | Hit Rate @ 5     |   0.3350 |               — |
| 3   | Hit Rate @ 10    |   0.7400 |               — |

KNNBasic successfully identifies at least one of a student's weak specialties for 74% of students at K=10. Precision@K is capped at ~0.60 because students only have 3–4 weak specialties. Perfect P@5 is impossible when there are fewer weak specialties than K.

## 10. RAG Pipeline & Benchmarks

### 10.1. Relevant Files

| #   | File                                      | Purpose                                               |
| --- | ----------------------------------------- | ----------------------------------------------------- |
| 1   | `notebooks/04_rag_pipeline.ipynb`         | pilot run (50 questions, Qwen3-4B)                    |
| 2   | `notebooks/07_evaluation_benchmark.ipynb` | full ablation grid                                    |
| 3   | `models/benchmarks.json`                  | all run results                                       |
| 4   | `config/models.json`                      | `benchmark_combinations` array defines the exact grid |

### 10.2. Benchmark combinations

This is defined in `config/models.json > benchmark_combinations`.

| #     | Embedding Model          | LLM                   | RAG   | n_eval  | Accuracy | Delta     |
| ----- | ------------------------ | --------------------- | ----- | ------- | -------- | --------- |
| 1     | Qwen3-Embedding-0.6B     | Qwen3-4B              | ✗     | 50      | 42%      | —         |
| 2     | Qwen3-Embedding-0.6B     | Qwen3-4B              | ✓     | 50      | 38%      | -4pp      |
| 3     | Qwen3-Embedding-0.6B     | Qwen3-4B-Thinking     | ✗     | 100     | 31%      | —         |
| 4     | Qwen3-Embedding-0.6B     | Qwen3-4B-Thinking     | ✓     | 100     | 32%      | +1pp      |
| 5     | Octen-Embedding-0.6B     | Qwen3-4B-Thinking     | ✗     | 100     | 33%      | —         |
| **6** | **Octen-Embedding-0.6B** | **Qwen3-4B-Thinking** | **✓** | **100** | **44%**  | **+11pp** |

### 10.3. Finding

RAG effectiveness is embedding- and LLM-dependent. A general-purpose embedding (MiniLM) or standard LLM hurts performance. A biomedical embedding (Octen, RTEB Healthcare rank #15) paired with a reasoning-capable LLM (Qwen3-4B-Thinking-2507) gives +11pp. NER rewriting is necessary but not sufficient — the LLM must also be capable of using the retrieved context.

## 11. FastAPI Backend

### 11.1. Relevant Files

| #   | File         | Purpose                          |
| --- | ------------ | -------------------------------- |
| 1   | `src/api.py` | FastAPI app                      |
| 2   | `run_api.py` | server entrypoint with CLI flags |

### 11.2. Endpoints

| #   | Method | Path          | Description                                        |
| --- | ------ | ------------- | -------------------------------------------------- |
| 1   | GET    | `/health`     | Service health, backend info, feature flags        |
| 2   | POST   | `/webhook`    | Dialogflow ES webhook (two-turn async RAG pattern) |
| 3   | POST   | `/chat`       | Direct EMMA query: full RAG, no Dialogflow timeout |
| 4   | POST   | `/query`      | Developer testing endpoint                         |
| 5   | GET    | `/conditions` | Lists evaluation-domain conditions                 |

### 11.3. Two-Turn Async Pattern

Dialogflow ES enforces a 5-second response timeout. LLM inference takes 8–90 seconds. The webhook immediately returns an acknowledgment ("Looking that up...") and fires RAG as a background task. On the next user message, it delivers the completed answer. This gives real RAG responses through Dialogflow with zero timeouts.

## 12. Setup

### 12.1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Ollama](https://ollama.com) — for local LLM inference (optional but recommended)
- [ngrok](https://ngrok.com) — for exposing the API to Dialogflow (optional)

### 12.2. Install

```bash
git clone https://github.com/jaxendutta/emma.git
cd emma

# Unix / WSL
bash scripts/setup.sh

# Windows PowerShell
scripts\setup.ps1
```

The setup script:

1. Creates a `.venv` and installs all dependencies via `uv sync`
2. Installs both SpaCy biomedical models (`en_core_sci_md` + `en_ner_bc5cdr_md`)
3. Registers the Jupyter kernel (`emma`)
4. Verifies the `src` package is importable

### 12.3. Environment variables

Copy `.env.example` to `.env` and fill in as needed:

```bash
cp .env.example .env
```

```env
HF_TOKEN=hf_your_token_here     # required only for gated models
EMMA_USE_RAG=true               # enable RAG pipeline in the API
EMMA_MODEL_ID=qwen3-4b          # override default LLM (optional)
EMMA_OLLAMA_URL=http://localhost:11434
```

### 12.4. Pull the LLM (Ollama)

```bash
ollama pull qwen3:4b-thinking-2507-q4_K_M   # champion model (~2.5 GB)
ollama pull qwen3:4b                         # standard variant (~2.5 GB)
```

### 12.5. Open notebooks

```bash
uv run jupyter notebook notebooks/
```

Select the `EMMA` kernel when prompted.

## 13. Running the API Locally

### 13.1. Start the server

```bash
# Static knowledge only (no LLM, instant startup)
uv run python run_api.py

# Full RAG pipeline
uv run python run_api.py --rag

# Specify model and port
uv run python run_api.py --rag --model qwen3-4b --port 8000

# Production mode (no auto-reload)
uv run python run_api.py --rag --no-reload
```

The server starts at `http://localhost:8000`. Check `http://localhost:8000/health` to confirm it's running and inspect backend status.

### 13.2. Expose to Dialogflow via ngrok

Dialogflow requires a public HTTPS URL to reach your webhook. ngrok creates a secure tunnel from a public URL to your local server.

**1. Install ngrok and authenticate:**

```bash
# Install: https://ngrok.com/download
ngrok authtoken YOUR_NGROK_TOKEN    # get token at dashboard.ngrok.com
```

**2. In a separate terminal, start the tunnel:**

```bash
ngrok http 8000
```

ngrok will print a URL like `https://abc123.ngrok-free.app`. Copy it.

**3. Update Dialogflow:**

- Go to your Dialogflow ES agent → Fulfillment → Webhook
- Set URL to: `https://abc123.ngrok-free.app/webhook`
- Save, then re-enable webhook fulfillment on each intent

> [!NOTE]
> The ngrok URL changes every time you restart ngrok on the free plan. You'll need to update Dialogflow each session. The paid plan ($10/month) gives you a static domain.
>
> **Session limits:** Free Colab cuts out after 12 hours (hard limit) and 90 minutes of inactivity. Run both `run_api.py` and the ngrok tunnel from your local machine if you want a longer-lived session. The Vercel static client works independently of the API:  Dialogflow's cloud servers handle the chatbot even when your local API is offline.

### 13.3. Test Directly (No DialogFlow)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the mechanism of anaphylaxis?", "use_rag": true}'
```

## 14. Key Design Decisions

| #   | Decision                                     | Rationale                                                                                                                                                                    |
| --- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Textbooks as RAG corpus                      | MedQA questions were written from these 18 textbooks — the ideal retrieval source. Faster and more reproducible than live PubMed querying.                                   |
| 2   | `en_ner_bc5cdr_md` for NER                   | Only ScispaCy model with typed DISEASE + CHEMICAL labels. `en_core_sci_md` produces a single generic ENTITY label — unsuitable for typed extraction.                         |
| 3   | Octen-Embedding-0.6B as production embedding | RTEB Healthcare rank #15, ablation validated (+11pp RAG delta). Qwen3-Embedding is the build default due to earlier availability.                                            |
| 4   | Qwen3-4B-Thinking-2507 as production LLM     | Medmarks rank #33 in tiny model category. The thinking variant is required for RAG — the standard Qwen3-4B cannot effectively use retrieved context (ablation result: -4pp). |
| 5   | Separation of concerns                       | The ML pipeline makes deterministic routing decisions; the LLM generates explanations only. Routing is auditable and fast; generation is where latency lives.                |
| 6   | `benchmark_combinations` in `models.json`    | Explicit grid declaration avoids accidental cross-product runs. The ablation loop iterates exactly what is declared — no more, no less.                                      |
| 7   | Stratified CV subsampling                    | 20k stratified sample for model selection; champion retrained on full 179k corpus. The CV-to-holdout gap (0.54 → 0.69 F1) is expected and documented.                        |
| 8   | Score thresholding + confidence bands        | Chunks below 0.40 are dropped; 0.40–0.55 flagged as "low confidence"; 0.55–0.70 as "medium". The LLM is instructed not to rely on low-confidence sources.                    |
| 9   | Two-turn async webhook                       | Dialogflow ES has a 5-second deadline; LLM inference takes 8–90 seconds. The webhook returns an acknowledgment immediately and delivers the RAG answer on the next turn.     |

## References

1. Rezaei, M. R., Saadati Fard, R., Parker, J. L., Krishnan, R. G., & Lankarany, M. (2025). Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge. In *Findings of the Association for Computational Linguistics: EMNLP 2025* (pp. 12682–12701). ACL.

2. Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). ScispaCy: Fast and robust models for biomedical natural language processing. In *Proceedings of the 18th BioNLP Workshop* (pp. 319–327). ACL.
