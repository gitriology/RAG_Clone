# RAGar — MS-ARC: Adaptive Multi-State Retrieval with Evidence Graph Reasoning

A secure, domain-adaptive, offline-capable **Retrieval-Augmented Generation (RAG)** framework built around a
**Multi-State Adaptive Retrieval Controller (MS-ARC)**. Instead of retrieving a fixed Top-K for every query,
MS-ARC retrieves incrementally (5 → 10 → 15 → 20), measures the quality of the evidence collected so far, and
**stops as soon as the evidence is sufficient** — then validates the evidence, synthesises a grounded answer and
reports a calibrated confidence.

> This README documents the project end to end: initial implementation → hybrid retrieval → reranking →
> hallucination control → confidence scoring → frontend → MS-ARC novelty and optimisations → Groq LLM synthesis →
> Firebase auth/history → Dockerisation → AWS EC2 + Nginx + HTTPS → Vercel.

---

## Table of contents

1. [Overview](#1-overview)
2. [Research gap and novelty](#2-research-gap-and-novelty)
3. [System architecture](#3-system-architecture)
4. [Technology stack (proposed vs implemented)](#4-technology-stack-proposed-vs-implemented)
5. [Repository structure](#5-repository-structure)
6. [Knowledge base and dataset](#6-knowledge-base-and-dataset)
7. [Implementation walkthrough (initial build)](#7-implementation-walkthrough-initial-build)
8. [MS-ARC: the adaptive retrieval controller](#8-ms-arc-the-adaptive-retrieval-controller)
9. [Optimisation log (#4 – #41)](#9-optimisation-log-4--41)
10. [LLM integration (Groq)](#10-llm-integration-groq)
11. [Firebase authentication and chat history](#11-firebase-authentication-and-chat-history)
12. [Local setup and run](#12-local-setup-and-run)
13. [API reference](#13-api-reference)
14. [Testing and benchmarks](#14-testing-and-benchmarks)
15. [Deployment](#15-deployment)
16. [Troubleshooting](#16-troubleshooting)
17. [Known gaps and notes](#17-known-gaps-and-notes)
18. [Citation](#18-citation)

---

## 1. Overview

Conventional RAG retrieves the same number of chunks (Top-K) for every query. A fixed small K gives
*insufficient* evidence for complex questions; a fixed large K dilutes the prompt with redundant or contradictory
text and wastes CPU/GPU cycles — a real problem for offline, air-gapped and resource-constrained deployments.

MS-ARC turns retrieval into a **feedback-driven control loop**:

```
Query → Hybrid retrieval (FAISS + BM25) → Cross-Encoder rerank → Evidence Graph
      → Evidence State Vector (ESV) → Evidence Sufficiency Criterion (ESC)
      → Retrieve again  OR  Generate → Grounded answer + confidence
```

### Key results (reported in the paper)

Evaluated on a curated domain corpus of **2,013 chunks** and **50 information-seeking queries** across five
domains (Legal, Healthcare, Education, Finance, Space Missions). Retrieval Score = mean Recall@K and Ranking
Score = mean MRR@K over K ∈ {5, 10, 15}; values are mean ± sample standard deviation across queries.

| Approach | Retrieval score | Ranking score | Avg. retrieval depth | Latency (ms) |
|---|---|---|---|---|
| Fixed retrieval | 0.900 ± 0.316 | 0.767 ± 0.387 | 20.0 | 182 |
| Complexity-adaptive retrieval | 0.900 ± 0.316 | 0.767 ± 0.387 | 13.8 | 156 |
| **MS-ARC** | **1.000 ± 0.000** | **0.850 ± 0.242** | **11.4** | **139** |

That is a **43 % reduction in retrieval depth** (20.0 → 11.4) and a **23.6 % reduction in latency** versus fixed
retrieval (10.9 % versus complexity-adaptive retrieval).

Ablation (paper Fig. 6, overall evaluation score): Full MS-ARC **0.89**, without stability 0.84, without
agreement 0.82, without confidence margin 0.85. Confidence convergence (paper Fig. 5) rises 0.54 → 0.69 → 0.78
→ 0.86 → 0.90 over five retrieval stages, then flattens — the diminishing-returns region the ESC exploits.

> These figures come from the manuscript's evaluation corpus. See [§6](#6-knowledge-base-and-dataset) for how the
> knowledge base shipped in the repository relates to it.

---

## 2. Research gap and novelty

### Research gap (from the project proposal)

| # | Gap | Statement |
|---|---|---|
| 1 | No end-to-end reliable pipeline | Existing RAG work optimises single components (retrieval *or* reranking *or* hallucination) and lacks one integrated architecture. |
| 2 | Weak multi-layer hallucination control | Methods validate either retrieval *or* output, rarely both. |
| 3 | No explicit confidence scoring | Answers come with sources but not a quantified reliability score. |
| 4 | Leaky domain restriction | Domain-specific systems still allow general LLM knowledge to leak into answers. |
| 5 | Secure/offline deployment ignored | Most research assumes cloud APIs and ignores privacy-sensitive environments. |

### Novelty

**Primary research contribution — MS-ARC.** Retrieval depth is treated as a runtime optimisation problem
instead of a fixed pre-processing step. Minimise K subject to `Answer quality ≥ τ`, with

```
K = f(Q, A, M, S, C)
```

| Symbol | Signal | Meaning |
|---|---|---|
| Q | Query complexity | Query length, entity signal, technical density, multi-hop language |
| A | Hybrid retrieval agreement | Overlap between dense (FAISS) and sparse (BM25) candidate sets |
| M | Cross-Encoder margin | `ΔS = S₁ − S₂`, gap between the best and second-best reranked document |
| S | Retrieval stability | Whether the top-ranked set stays the same as depth increases |
| C | Confidence convergence | ε-convergence — stop when extra retrieval no longer improves confidence |

Three structures make the controller explicit and auditable:

* **Evidence Graph** `G = (V, E)` — nodes are evidence passages/sentences, edges model relationships between them.
* **Evidence State Vector (ESV)** — a normalised multi-dimensional summary of the current evidence
  (coverage, authority, contradiction, freshness, retrieval confidence, computational cost).
* **Evidence Sufficiency Criterion (ESC)** — a weighted score of the ESV compared with a threshold to decide
  *Generate* or *Retrieve again*.

**Supporting contributions (system level).**

1. Hybrid retrieval (dense + BM25) for domain-specific terminology.
2. Multi-stage hallucination mitigation: hybrid retrieval → context validation → reranking → controlled
   prompting → output validation.
3. Unified confidence score combining retrieval, reranking and validation signals.
4. Strict domain-restricted generation — answer only from retrieved context, otherwise *"I don't know based on
   the available knowledge base."*
5. Secure, offline-capable deployment (private vector index, local models, optional hosted LLM).
6. Modular, domain-adaptive design — the pipeline is unchanged across domains; only the knowledge base changes.

### How this differs from black-box tools

| Aspect | Black-box assistants | This system |
|---|---|---|
| Pipeline | Internal / opaque | Every stage explicit and inspectable (retrieval → rerank → validate → score) |
| Domain boundary | May reason beyond documents using pre-trained knowledge | Hard rule: not in the context → "I don't know" |
| Trust | Shows sources, no numeric trust | Numeric pipeline, retrieval and answer confidence |

> *If asked "isn't this already done?":* individual components exist; the contribution is integrating them into a
> controlled, domain-restricted, trust-aware pipeline whose **retrieval depth is decided by the evidence state**.

---

## 3. System architecture

### Pipeline (paper Fig. 1)

```
 1  INPUT            User query
 2  DUAL RETRIEVAL   Dense (FAISS, BGE)  ‖  Sparse (BM25)         → Top-Nd / Top-Ns
 3  HYBRID FUSION    Min-Max (default) or RRF                     → Top-Nf fused candidates
 4  MS-ARC CONTROL   Query complexity · Agreement · Margin · Stability · Confidence
                     → selected retrieval depth K  (iterative 5 → 10 → 15 → 20)
 5  FINAL SELECTION  Cross-Encoder reranking → Top-K documents
 6  OUTPUT           Evidence-grounded answer (+ confidence, sources, evidence graph/state)
```

### Production request path

```
React (Vite) ──POST /api/query──► FastAPI (backend/main.py)
                                   │
                                   ├─ Groq query planner (follow-up resolution, multi-topic split)   [optional]
                                   │
                                   └─ for each planned search query: run_pipeline()
                                        ├─ MS-ARC: complexity → adaptive plan → iterative hybrid retrieval
                                        │          (agreement · margin · stability · decision · evidence state)
                                        ├─ Cross-Encoder scores reused (no second rerank)
                                        ├─ Context validation (MiniLM)
                                        ├─ Top-K document selection
                                        ├─ Question-aware sentence selection  (Phase 10.1)
                                        ├─ Evidence fusion (graph-aware ordering)  (Phase 10.2)
                                        ├─ Answer validation + confidence calibration
                                        └─ Evidence Graph + Evidence State attached to the result
                                   │
                                   ├─ Merge evidence/sources across planned queries
                                   └─ Groq synthesis (grounded, Markdown)  → falls back to the extractive answer
```

### Mathematical formulation

Hybrid score (Eq. 1) — `λ = 0.6` dense, `1 − λ = 0.4` sparse:

$$S(d,q)=\lambda\,S_{dense}(d,q)+(1-\lambda)\,S_{sparse}(d,q)$$

Evidence graph (Eq. 2) and Evidence State Vector (Eq. 3):

$$G=(V,E) \qquad ESV=[C,A,X,F,R,P]$$

`C` coverage · `A` source authority · `X` contradiction · `F` freshness · `R` retrieval confidence ·
`P` computational cost. Each component is normalised to [0, 1]; higher C, A, F, R is better, higher X, P is worse.

Evidence sufficiency score (Eq. 4) and decision rule (Eq. 5):

$$Q_{ESV}=w_C C+w_A A+w_F F+w_R R-w_X X-w_P P$$

$$D=\begin{cases}\text{Generate}, & Q_{ESV}\ge\tau_Q\\[2pt] \text{Retrieve Again}, & Q_{ESV}<\tau_Q\end{cases}$$

Adaptive objective: `min K  subject to  Answer quality ≥ τ`.

---

## 4. Technology stack (proposed vs implemented)

The original proposal listed a broad stack; this table records what was actually built.

| Layer | Proposed | Implemented |
|---|---|---|
| Frontend | React, Next.js, Tailwind, TypeScript, Axios | React 19 + Vite, Tailwind, Axios, react-markdown (KaTeX, highlight.js), Lucide |
| API | Python, FastAPI, Uvicorn | FastAPI, Uvicorn, Pydantic |
| Embeddings | Sentence Transformers | `BAAI/bge-base-en-v1.5` (768-d, normalised) |
| Vector index | FAISS | FAISS (`IndexHNSWFlat`, inner product; `IndexFlatIP` supported) |
| Lexical retrieval | BM25 | `rank-bm25` with a persistent, fingerprint-validated cache |
| Reranker | Sentence-Transformers cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Validation model | cosine similarity | `all-MiniLM-L6-v2` (shared instance) |
| Complexity analysis | spaCy | Lightweight regex/token features by default; spaCy `en_core_web_sm` optional |
| Evidence graph | — | NetworkX |
| Document processing | PyPDF2, pdfplumber, unstructured | PyMuPDF (`pymupdf`), NLTK |
| LLM | LLaMA 3 / Mistral via Ollama | Groq-hosted `openai/gpt-oss-120b` (final synthesis only), extractive fallback |
| Auth / history | JWT + PostgreSQL / MongoDB | Firebase Authentication (Google Sign-In) + Cloud Firestore (frontend-side) |
| Hallucination tooling | guardrails-ai, ragas | Custom context/answer validators and calibrated confidence |
| Deployment | Docker, Kubernetes | Docker on AWS EC2 behind Nginx (+ Let's Encrypt) · Vercel frontend |

---

## 5. Repository structure

```
RAG-optimisation/                    # repository root (branch used for deployment: `deployable`)
├── Dockerfile                       # production image (CPU-only PyTorch, builds retrieval artifacts)
├── .dockerignore
├── .gitignore
├── check_faiss_id_alignment.py      # FAISS row-id ↔ dataset-id alignment check
├── optimization_14/15/16_results.json
├── data/
│   ├── raw/<domain>/*.pdf           # source PDFs (git-ignored)
│   └── processed/all_domains.json   # the committed knowledge base used at runtime
├── backend/
│   ├── main.py                      # FastAPI app: /api/health, /api/query
│   ├── llm_service.py               # Groq query planner + grounded answer synthesis
│   ├── requirements.txt
│   ├── models/model_registry.py     # shared BGE / MiniLM / CrossEncoder / spaCy singletons
│   ├── data_pipeline/               # PDF → clean → chunk → metadata → JSON
│   │   ├── ingestion/load_pdf.py
│   │   ├── cleaning/clean_text.py
│   │   ├── chunking/chunker.py
│   │   ├── metadata/tagger.py
│   │   ├── utils/{normalise_json,save_json}.py
│   │   └── run_pipeline.py
│   ├── retrieval/
│   │   ├── dense/embedder.py        # BGE embeddings (batch + query encoding)
│   │   ├── lexical/bm25.py          # shared tokenizer + persistent BM25 cache
│   │   ├── index/{faiss_index,build_index}.py
│   │   ├── hybrid/{hybrid_search,fusion,rrf}.py
│   │   ├── utils/{load_data,save_embeddings,load_embeddings}.py
│   │   ├── query_matching.py        # entity/reference/intent helpers
│   │   └── run_retrieval.py         # retrieve(): MS-ARC entry for the pipeline
│   ├── ms_arc/                      # the adaptive retrieval controller
│   │   ├── run_msarc.py             # iterative controller loop
│   │   ├── complexity/analyzer.py
│   │   ├── agreement/agreement.py
│   │   ├── confidence/margin.py
│   │   ├── stability/stability.py
│   │   ├── decision/decision_engine.py
│   │   ├── retrieval/{retrieve,adaptive_policy,candidate_controller}.py
│   │   └── state/{retrieval_state,retrieval_signals}.py
│   ├── evidence_graph/              # nodes, edges, clustering, analytics, ranking, serialization
│   ├── evidence_state/              # features, normalisation, weighting, incremental state, serialization
│   ├── reranker/
│   │   ├── model/reranker.py
│   │   ├── utils/context_selector.py
│   │   └── scoring/{run_rerank.py,confidence/confidence_score.py}
│   ├── generation/
│   │   ├── answer_generator.py      # Phase 10.1 semantic sentence selection
│   │   ├── evidence_fusion.py       # Phase 10.2 graph-aware evidence fusion
│   │   ├── answerability.py
│   │   ├── prompt_builder.py
│   │   ├── validation/{context_validator,embedding_cache}.py
│   │   └── guards/answer_validator.py
│   ├── evaluation/                  # Min-Max vs RRF fusion benchmark + labelled dataset
│   ├── benchmarks/                  # per-optimisation micro-benchmarks
│   └── test_*.py                    # 26 pytest modules
└── frontend/
    ├── package.json · vite.config.js · vercel.json · tailwind.config.js
    └── src/
        ├── App.jsx · App.css · main.jsx
        ├── components/{Login,Sidebar,Header,ChatWindow,MessageBubble,InputBox,RightPanel,LogoMark}.jsx
        ├── context/AuthContext.jsx
        ├── firebase/{config,auth}.js
        └── services/{api,chatService}.js
```

---

## 6. Knowledge base and dataset

* The runtime knowledge base is **`data/processed/all_domains.json`** — a JSON list of chunk records. This is the
  only data file the deployed backend needs; the raw PDFs are not required at runtime.
* Each record carries: `id`, `text`, `domain`, `source`, `type`, `document_id`, `page_number`, `section`,
  `timestamp`, `source_path`, `source_file_hash`, `chunk_index`, `page_start`, `page_end`, `word_count`,
  `chunk_quality`, `text_hash`, `preprocessing_version`.
* **Rule:** whenever `all_domains.json` changes, the embeddings, FAISS index and BM25 cache must be rebuilt
  (see [§12](#12-local-setup-and-run)). The Docker build does this automatically.

> **Check which dataset you ship.** The manuscript and the benchmark result files (`optimization_15/16_results.json`
> report `ntotal: 2013`, dimension 768) refer to the **2,013-chunk, five-domain** evaluation corpus
> (Legal, Healthcare, Education, Finance, Space Missions). The `all_domains.json` inside the
> `RAG_Clone-llm-integration` snapshot is a smaller **690-chunk** knowledge base
> (`indian_constitution` 610, `computer_science` 63, `healthcare` 11, `history` 6; sources include
> `indian_constitution.pdf`, `os.pdf`, `dbms.pdf`, `iot.pdf`, `parallel_programming.pdf`, `who_vaccination.pdf`,
> `ancient_india.pdf`), which matches the LLM-integration demo (e.g. IoT / DBMS questions). Reproduce the paper's
> numbers only with the 2,013-chunk corpus; state the corpus used whenever you report results.

Recommended domains for the proposal (public, non-sensitive sources): Space/Aerospace (ISRO, NASA), Legal/Government
policy (Constitution, GFR, circulars), Healthcare (WHO, NIH — never patient data), Finance (RBI, SEBI), Education
(NPTEL, textbooks, university material).

---

## 7. Implementation walkthrough (initial build)

The project was built phase by phase; each phase was tested on its own before the next one started
(*"keep scripts separate, print sample outputs, don't jump to embeddings yet"*).

### Phase 0 — Environment

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      |  macOS/Linux:  source .venv/bin/activate
pip install pymupdf tqdm
```

### Phase 1 — Data collection and preparation

Goal: **PDF → text → clean → chunk → metadata → JSON**.

Initial scaffold (`data/<domain>/{raw,extracted,cleaned,processed}` + `scripts/{extract,clean,chunk,metadata}.py`):

```python
# extract.py — page-wise text extraction
import fitz  # pymupdf
def extract_pdf(input_path, output_path):
    doc = fitz.open(input_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    open(output_path, "w", encoding="utf-8").write(full_text)

# clean.py — collapse whitespace
import re
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# chunk.py — word-based chunks with overlap (300–500 tokens, 50–100 overlap recommended)
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# metadata.py — attach domain/source metadata to every chunk
def add_metadata(chunks, doc_name):
    return [{"chunk_id": f"{doc_name}_{i}", "text": c, "domain": "legal", "document_name": doc_name}
            for i, c in enumerate(chunks)]
```

The scaffold then grew into the production pipeline `backend/data_pipeline/run_pipeline.py`, which adds:
layout-aware extraction (PyMuPDF), list-based page aggregation (#38), metadata-aware page-number/header cleaning
(#39), sentence-aware chunking with overlap (`DEFAULT_CHUNK_SIZE=500`, `DEFAULT_OVERLAP=100` words,
`MIN_CHUNK_WORDS=45`, `MAX_CHUNK_WORDS=600` — chosen by experiment #40), and enriched metadata with SHA-256
content hashes (#41). Domain is inferred from the first directory under `data/raw/`.

```bash
# data/raw/<domain>/*.pdf  →  data/processed/all_domains.json
python -m backend.data_pipeline.run_pipeline
# optional explicit paths
python -m backend.data_pipeline.run_pipeline --input data/raw --output data/processed/all_domains.json
```

Planned metadata model: `domain` (technical/admin), `source` (manual/policy), `department`, `confidentiality`.

### Phase 2 — Hybrid retrieval

Goal: **Query → Dense (FAISS) → BM25 → Hybrid merge → Top-K**.

Initial dependencies and code layout (`backend/retrieval/{dense,lexical,hybrid,index,utils}`):

```bash
pip install sentence-transformers faiss-cpu rank-bm25 numpy
```

The first version used `all-MiniLM-L6-v2`, `IndexFlatL2` and a set-union merge. It evolved into the production
retriever:

| Component | Production behaviour |
|---|---|
| Dense | `BAAI/bge-base-en-v1.5`, normalised 768-d embeddings, batch encoding (`encode_queries`, #16) |
| Index | FAISS `IndexHNSWFlat` (M=32, efConstruction=200, efSearch=64), inner product = cosine on normalised vectors; `flat` available |
| Sparse | BM25Okapi, shared lowercase/word tokenizer for documents *and* queries, persistent cache validated by SHA-256 corpus fingerprint (#14) |
| Fusion | **Min-Max** weighted (0.6 dense / 0.4 BM25) — default; **RRF** (k=60) as an ablation baseline (#6) |
| Candidate pool | Adaptive 10–40 candidates from complexity, query type, dense/sparse agreement, stability |
| Extras | LRU query cache (128 entries), lexical-anchor recovery for exact references/acronyms (e.g. "Article 12") |

The persisted artifacts (generated, never committed):

```
backend/retrieval/cache/embeddings.npy
backend/retrieval/index/faiss.index
backend/retrieval/index/bm25_cache.pkl
```

### Phase 3 — Reranking and grounding

Two-stage retrieval: **Stage 1** fast bi-encoder + BM25 recall, **Stage 2** deep Cross-Encoder scoring that reads
query and passage together.

```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents):
    if not documents:
        return []
    scores = model.predict([(query, d["text"]) for d in documents])
    for d, s in zip(documents, scores):
        d["rerank_score"] = float(s)
    return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
```

`select_top_k(ranked_docs, k=3)` (in `reranker/utils/context_selector.py`) keeps only the best documents so the
generator gets a small, clean context. In production the Cross-Encoder runs **once**, inside MS-ARC, and its scores
are reused downstream (#4, #29).

### Phase 4 — Hallucination prevention (multi-layer verification)

```
Retrieval → Reranking → Context validation → Top-K selection → Extractive answer → Answer validation
```

* **Context validation** (`generation/validation/context_validator.py`): cosine similarity between the query and each
  retrieved chunk using MiniLM; initial threshold **0.3**; stores `context_score` on each document.
* **Controlled, extractive answer construction**: the answer is assembled from real source sentences rather than free
  generation (hallucination-safe, deterministic, works offline).
* **Answer validation** (`generation/guards/answer_validator.py`): similarity between the answer and the documents;
  initial threshold **0.4**; returns `{"is_valid": bool, "confidence": float}`.

> Describe this layer in the paper as **similarity-based answer support**, not as proof of factual entailment.

### Phase 5 — Confidence scoring

First formulation (kept as the conceptual baseline):

```
Confidence = 0.4 · normalised_rerank + 0.3 · avg_context_score + 0.3 · answer_validation_score
normalised_rerank = clamp((avg_rerank + 3) / 6, 0, 1)      # maps cross-encoder logits [-3, 3] → [0, 1]
```

The production scorer (`reranker/scoring/confidence/confidence_score.py`, calibration
`evidence_aware_v5_answer_form`) evolved this into bounded evidence signals with conservative gates — rank
separation of the Cross-Encoder, document/sentence context support, answer agreement, query grounding, evidence
quality/contamination and source consistency. It deliberately does **not** present logits as probabilities.
The API exposes `pipeline_confidence`, `retrieval_confidence` and `answer_confidence` separately.

### Phase 6 — Frontend and API integration

A React (Vite) client calls the single master endpoint:

```
POST /api/query     {"query": "What is vaccination?"}
```

Response (abridged): `answer`, `confidence`, `domain`, `sources[]`, `pipeline_version`, `meta` (pipeline steps and
retrieval diagnostics). CORS is enabled for the browser client, API failures are handled in the UI, and an empty
retrieval returns a safe fallback (`"I don't know based on the available knowledge base."`, confidence 0.0).

Planned (proposal) vs delivered: domain classifier, JWT auth and database-backed filters from the proposal were
replaced by metadata-driven `domain` fields, Firebase Authentication and Firestore (see [§11](#11-firebase-authentication-and-chat-history)).

---

## 8. MS-ARC: the adaptive retrieval controller

Implemented in `backend/ms_arc/` and driven by `run_msarc()`.

### 8.1 Signals

| Signal | Module | Definition |
|---|---|---|
| Query complexity | `complexity/analyzer.py` | `0.30·length + 0.20·entity + 0.25·technical-density + 0.25·multi-hop` (length normalised over 2–30 tokens, entity signal over 0–6). Buckets: `simple` (<0.30, K=3), `medium` (<0.55, K=5), `complex` (<0.75, K=8), `very_complex` (K=10); "compare" intents raise the recommendation to 10, "why/how/explain" to 8. Lightweight regex mode by default; `MSARC_COMPLEXITY_NER=spacy` re-enables NER. |
| Hybrid agreement (A) | `agreement/agreement.py` | Score-weighted intersection ÷ score-weighted union of the dense and sparse candidate sets. |
| Cross-Encoder margin (M) | `confidence/margin.py` | `(S₁ − S₂) / (|S₁| + ε)` over reranked documents; scores are cached so a document is never scored twice. |
| Stability (S) | `stability/stability.py` | Jaccard overlap between successive iteration rankings (no extra FAISS/BM25 searches). |
| Decision | `decision/decision_engine.py` | `confidence = 0.40·A + 0.35·M + 0.25·S`; ≥0.75 `ACCEPT`, ≥0.55 `EXPAND_TOPK`, ≥0.35 `RERANK_AGAIN`, else `RETRIEVE_AGAIN`. |

### 8.2 Adaptive retrieval policy (`retrieval/adaptive_policy.py`)

| Parameter | Default |
|---|---|
| `initial_k` | 5 |
| `expansion_step` | 5 (schedule 5 → 10 → 15 → 20) |
| `max_k` | 20 (ceiling scales with query complexity) |
| `confidence_threshold` | 0.75 |
| `convergence_epsilon` | 0.02 |

### 8.3 The loop

```
plan = policy.plan(query_complexity, recommended_topk)
K = plan.initial_k
repeat:
    retrieve(K)                          # hybrid FAISS + BM25, adaptive candidate pool
    agreement → margin (Cross-Encoder, cached) → stability → decision
    update cheap incremental Evidence State                       (#36)
    stop if  evidence_score ≥ threshold
          or retrieval confidence ≥ threshold
          or confidence gain ≤ ε            (confidence converged)
          or K ≥ max_k
    if still ambiguous: build lazy Evidence Graph + full Evidence State
          and stop if it is sufficient                            (#37)
    otherwise K ← K + 5
```

Each iteration is logged (`iteration_history`) with `stop_reason` ∈ `confidence_threshold`,
`evidence_state_sufficient`, `full_evidence_state_sufficient`, `confidence_converged`, `max_k_reached`, `expand`.

### 8.4 Evidence Graph and Evidence State

* **Evidence Graph** (`backend/evidence_graph/`): nodes are evidence items; edges are built from one embedding
  matrix and a cosine-similarity matrix (similarity threshold 0.55, at most 3 neighbours per node) instead of
  O(N²) model calls. Clustering, centrality (PageRank, betweenness, eigenvector, …), coherence, consensus, quality
  and validation are available; heavy analytics run lazily (#34, #35).
* **Evidence State** (`backend/evidence_state/`): features grouped as Retrieval, Graph, Quality, Ranking and
  Diagnostics; per-feature normalisation (not per-query min-max); weighted vector → `evidence_score`.
  `EVIDENCE_WEIGHT_MODE` selects `manual` (default) or tuned weights
  (`weighting/tuned_feature_weights.json`, #23).

---

## 9. Optimisation log (#4 – #41)

A code review of the first working system identified duplicated work and an MS-ARC layer that computed signals
without actually looping. The fixes were done in priority order (model loading → duplicate Cross-Encoder →
repeated retrieval → repeated embeddings → real iterative control), each in its own commit/branch and each with a
benchmark or test, so every claim can be backed by a baseline vs optimised measurement.

| # | Change | Where |
|---|---|---|
| 1 | Central `ModelRegistry`: BGE, MiniLM (shared by context + answer validators), Cross-Encoder and spaCy loaded once | `models/model_registry.py` |
| 4 / 29 | MS-ARC owns the one Cross-Encoder pass; downstream reranking reuses `state.reranked_results` | `ms_arc/confidence/margin.py`, `retrieval/run_retrieval.py` |
| 5 | Stability from nested top-K of a single retrieval instead of 3 extra hybrid searches | `ms_arc/stability/stability.py` |
| 6 | Adaptive candidate pool (10–40) from complexity, type, agreement, stability; **Min-Max vs RRF** ablation | `ms_arc/retrieval/candidate_controller.py`, `retrieval/hybrid/`, `evaluation/` |
| 14 | BM25 shared tokenizer + persistent, fingerprint-validated cache | `retrieval/lexical/bm25.py` |
| 15 | FAISS id-alignment check and Flat vs HNSW comparison | `check_faiss_id_alignment.py`, `benchmark_optimization_15.py` |
| 16 | Batch query encoding in one `model.encode()` call | `retrieval/dense/embedder.py` |
| 21 | Evidence Graph and Evidence State integrated into the production request path | `reranker/scoring/run_rerank.py`, `main.py` |
| 23 | Evidence-state feature weighting: equal vs manual vs tuned (positive-logistic) | `evidence_state/weighting/` |
| 24–26 | Lightweight complexity analysis (spaCy optional) and cost-aware adaptive plan | `ms_arc/complexity`, `ms_arc/retrieval/adaptive_policy.py` |
| 27 | Exact Flat-IP vs HNSW retrieval benchmark | `benchmarks/benchmark_retrieval_27.py` |
| 28 | **Genuinely iterative MS-ARC** (retrieve → evaluate → expand → stop) | `ms_arc/run_msarc.py` |
| 30 | Validation-embedding reuse (shared MiniLM, batch + normalised) | `generation/validation/` |
| 32 | One batched encode in the answer generator | `generation/answer_generator.py` |
| 33 | Evidence-graph edges from a similarity matrix over precomputed embeddings | `evidence_graph/edge_builder.py` |
| 34 / 35 | Lazy graph analytics (only when the state is ambiguous) | `evidence_graph/graph_builder.py`, `analytics/` |
| 36 | Cheap incremental Evidence State updated every iteration | `evidence_state/incremental.py` |
| 37 | Full Evidence State used as a retrieval-control signal only when still ambiguous | `ms_arc/run_msarc.py` |
| 38 | PDF text aggregated with list + single join | `data_pipeline/ingestion/load_pdf.py` |
| 39 | Metadata-aware page-number/header cleaning | `data_pipeline/cleaning/clean_text.py` |
| 40 | Chunk-size experiment: 200/40, 300/60, 400/80, 500/100 → **500/100 selected** | `benchmarks/benchmark_chunking_40.py` |
| 41 | Chunk metadata enrichment (document id, page, chunk index, section, timestamp, SHA-256 hashes) | `data_pipeline/` |

Measured results stored in the repository:

| Experiment | Result |
|---|---|
| #14 BM25 persistent cache | median BM25 initialisation 0.352 s → 0.129 s (**−63 %**), median startup 0.419 s → 0.179 s (**−57 %**) |
| #15 / #16 HNSW vs Flat (2,013 vectors, 768-d, 5 labelled queries) | **≈3.4× faster search** with identical Recall/Precision/Hit-rate/MRR |
| #6 fusion ablation (Top-3, 10 labelled queries) | Min-Max: Recall 1.0, Precision 0.367, Hit-rate 1.0, MRR 0.85 · RRF: 0.8 / 0.30 / 0.8 / 0.633 → **Min-Max retained as default** |
| #40 chunk size | 500/100 selected as the winning configuration |

> Small labelled sets (5–10 queries) are sanity checks, not statistical proof; the 50-query evaluation in the paper is
> the headline result. Recommended before final claims: 100–300 representative queries, recording retrieval /
> rerank / generation / total latency, FAISS+BM25 call counts, Cross-Encoder calls, retrieved vs final K and answer
> quality for **Baseline RAG vs Optimised RAG vs Adaptive MS-ARC RAG**.

---

## 10. LLM integration (Groq)

Groq is used only as a **final answer-synthesis layer** over evidence the RAG pipeline already selected — it does
not replace retrieval or evidence selection.

| Item | Value |
|---|---|
| Provider / model | Groq API · `openai/gpt-oss-120b` (override with `GROQ_MODEL`) |
| Auth | `GROQ_API_KEY` (never commit it) |
| Module | `backend/llm_service.py` (called from `backend/main.py`) |

**Implementation steps**

1. Created `backend/llm_service.py` to isolate the model call and response logic.
2. Connected it from `backend/main.py` so the existing pipeline passes its selected evidence and context documents
   (`llm_context_documents`, carried through `reranker/scoring/run_rerank.py`) to the LLM.
3. Configuration through environment variables `GROQ_API_KEY` and `GROQ_MODEL`.
4. **Grounding prompt:** the model may use *only* the supplied evidence; conversation history is for reference
   resolution only; unsupported parts must be answered with `INSUFFICIENT_EVIDENCE`, which the service converts to
   *"I don't know based on the available knowledge base."*; code goes in fenced blocks; maths in LaTeX; internal
   pipeline details are never mentioned.
5. **Fallback:** any Groq failure (or a missing key) returns the existing extractive RAG answer instead of failing.
6. React rendering of Markdown, code and mathematics (`react-markdown`, `remark-gfm`, `remark-math`,
   `rehype-katex`, `rehype-highlight`).

**Extensions after the first integration**

* **Follow-up questions and reference resolution:** the last 6 turns are forwarded
  (`conversation_history` from `App.jsx` → `api.js` → `main.py`); a query-planning call rewrites "what are its
  types?" into a standalone query.
* **Multi-topic questions:** a request such as *"what is DBMS and define IoT"* is split (max 3) into independent
  searches; each runs through the unchanged RAG pipeline, then evidence and sources are merged before synthesis.
* **Conservative aggregation:** confidence is the minimum over the sub-queries; retrieval state shows the weakest
  agreement/margin/stability and the highest complexity/top-k.
* **Retrieval diagnostics in the UI** (right panel): agreement, complexity, margin, stability, recommended Top-K and
  retrieval confidence.

Tested locally with IoT questions and knowledge-base code rendering. Without `GROQ_API_KEY` the backend still
works and returns the extractive answer.

---

## 11. Firebase authentication and chat history

Design constraint: **identity and persistence live in the frontend; the FastAPI RAG backend is unchanged and does
not verify Firebase tokens.**

```
React ── Firebase Auth (Google Sign-In)  ── identifies the user
      ── Cloud Firestore                 ── stores conversations/messages
      ── FastAPI + RAG + Groq            ── answers (separate flow)
```

**Files (all in `frontend/src/`)**

| File | Responsibility |
|---|---|
| `firebase/config.js` | SDK initialisation from `VITE_FIREBASE_*`; exports `isConfigured` |
| `firebase/auth.js` | `signInWithGoogle`, `logout`, `subscribeToAuthChanges` |
| `context/AuthContext.jsx` | `AuthProvider` / `useAuth` — user, loading, auth errors |
| `components/Login.jsx` | Login screen; the chat UI is protected from unauthenticated access |
| `services/chatService.js` | create / list / update-title / delete conversations, add / replace / list messages |
| `components/Sidebar.jsx` | New chat, history list, open and delete conversations |

**Data model as implemented in `chatService.js`**

```
conversations/{conversationId}        { userId, title, createdAt, updatedAt }
  messages/{messageId}                { role, text, createdAt, title, confidence, sources,
                                        pipelineVersion, answerValid, retrievalConfidence,
                                        pipelineConfidence, answerConfidence, evidenceGraph,
                                        evidenceState, agreement, complexity, margin, stability,
                                        recommendedTopK, retrievalQueries, multiQuery }
```

(The original design sketch nested conversations under `users/{userId}/…`; the implementation uses a top-level
`conversations` collection filtered by `userId`.)

Behaviour: the user's question and the assistant's answer are saved after the RAG request; a Firestore failure must
not prevent a generated answer from being displayed; deleting a conversation deletes its messages in batches of 450.

**Firebase project setup**

1. Create a Firebase project → *Build → Authentication → Sign-in method* → enable **Google**.
2. Create a **Cloud Firestore** database.
3. Register a web app and copy its config into `frontend/.env.local` (see [§12](#12-local-setup-and-run)).
4. *Authentication → Settings → Authorized domains*: add `localhost` and your deployed frontend domain
   (e.g. `your-project.vercel.app`) so the Google popup works in production.
5. Publish Firestore security rules that isolate users, then verify with **two accounts** that neither can read or
   modify the other's conversations. The repository does **not** ship a `firestore.rules` file; a rule set matching
   the data model above:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /conversations/{conversationId} {
      allow create: if request.auth != null && request.resource.data.userId == request.auth.uid;
      allow read, update, delete: if request.auth != null && resource.data.userId == request.auth.uid;

      match /messages/{messageId} {
        allow read, write: if request.auth != null
          && get(/databases/$(database)/documents/conversations/$(conversationId)).data.userId == request.auth.uid;
      }
    }
  }
}
```

> **Status for reports:** the Firebase code above is present in the codebase, but the rules and the two-account
> isolation test are not part of the repository. Claim end-to-end Firebase functionality only after verifying it
> against your Firebase project.

---

## 12. Local setup and run

### 12.1 Prerequisites

| Tool | Notes |
|---|---|
| Git | `git --version` |
| Python 3.11 | matches the Docker image |
| Node.js 18+ and npm | for the Vite frontend |
| Docker Desktop (WSL 2 backend on Windows) | only for container testing |
| Groq API key | optional; without it you get extractive answers |

### 12.2 Clone and branch

```bash
git clone -b deployable https://github.com/<OWNER>/<REPO>.git
cd <REPO>
git branch          # expect: * deployable
git status          # expect a clean working tree
```

### 12.3 Backend environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

`backend/requirements.txt` (project-focused, not a full `pip freeze`):

```
fastapi
uvicorn[standard]
pydantic
numpy
pymupdf
nltk
sentence-transformers
faiss-cpu
rank-bm25
tqdm
spacy
scikit-learn
networkx
groq
python-dotenv
```

Create `.env` in the **repository root** (git-ignored; `llm_service.py` loads it with `python-dotenv`):

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
FRONTEND_ORIGIN=http://localhost:5173
```

### 12.4 Verify the knowledge base

```powershell
Test-Path data\processed\all_domains.json          # True
git ls-files data/processed/all_domains.json       # must print the path (tracked)
```

### 12.5 Build the retrieval artifacts

Run from the **repository root** (paths are relative):

```bash
# Only if you changed the raw PDFs:
python -m backend.data_pipeline.run_pipeline

# Always after the dataset changes:
python -m backend.retrieval.utils.save_embeddings     # → backend/retrieval/cache/embeddings.npy
python -m backend.retrieval.index.build_index         # → backend/retrieval/index/faiss.index (HNSW)

# Force BM25 regeneration (PowerShell)
Remove-Item backend\retrieval\index\bm25_cache.pkl -ErrorAction SilentlyContinue
# (bash)  rm -f backend/retrieval/index/bm25_cache.pkl

# Smoke test — the FAISS consistency checks must pass
python -c "import backend.ms_arc.retrieval.retrieve; print('retrieval import OK')"
```

Confirm the three artifacts exist, and that git ignores them:

```powershell
Test-Path backend\retrieval\cache\embeddings.npy
Test-Path backend\retrieval\index\faiss.index
Test-Path backend\retrieval\index\bm25_cache.pkl
git check-ignore -v backend\retrieval\cache\embeddings.npy
git check-ignore -v backend\retrieval\index\faiss.index
git check-ignore -v backend\retrieval\index\bm25_cache.pkl
```

### 12.6 Run the backend

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
# status: ok · pipeline_version: confidence-calibration-v3
```

Swagger UI: <http://127.0.0.1:8000/docs>

### 12.7 Run the frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local` (git-ignored):

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_FIREBASE_API_KEY=...
VITE_FIREBASE_AUTH_DOMAIN=...
VITE_FIREBASE_PROJECT_ID=...
VITE_FIREBASE_STORAGE_BUCKET=...
VITE_FIREBASE_MESSAGING_SENDER_ID=...
VITE_FIREBASE_APP_ID=...
```

```bash
npm run dev          # http://localhost:5173
npm run lint
npm run build
npm run preview      # confirm the production build
```

If `VITE_FIREBASE_*` are not all set, Firebase stays disabled (`isConfigured` is false) and the sign-in flow reports
that Firebase is not configured. Vite embeds `VITE_*` variables at build time — rebuild/redeploy after changing them.

---

## 13. API reference

### `GET /api/health`

```json
{ "status": "ok", "pipeline_version": "confidence-calibration-v3" }
```

### `POST /api/query`

Request:

```json
{
  "query": "What is immunization?",
  "top_k_documents": 3,
  "max_sentences": 3,
  "conversation_history": [ { "role": "user", "text": "..." }, { "role": "assistant", "text": "..." } ]
}
```

| Field | Type | Default | Constraint |
|---|---|---|---|
| `query` | string | — | min length 1 |
| `top_k_documents` | int | 3 | 1–20 (the shipped frontend sends 5) |
| `max_sentences` | int | 3 | 1–20 |
| `conversation_history` | list of `{role, text}` | `[]` | `role` ∈ `user`/`assistant`; last 6 turns used |

Response (abridged):

| Field | Meaning |
|---|---|
| `answer` | Final Markdown answer (Groq synthesis, or extractive fallback) |
| `confidence`, `pipeline_confidence` | Conservative pipeline confidence (minimum over sub-queries) |
| `retrieval_confidence`, `retrieval_state` | MS-ARC signals: agreement, complexity, margin, stability, recommended_top_k, decision, decision_reason |
| `domain`, `sources[]` | Domain of the primary result; every source document used |
| `pipeline_version` | Pipeline/calibration version |
| `evidence_graph`, `evidence_state` | Graph statistics and Evidence State summary |
| `selected_evidence`, `retrieved_documents` | Evidence sentences and (up to 6) context documents |
| `meta` | `status` (`success` / `low_confidence` below 0.40 / `no_answer`), `standalone_query`, `search_queries`, `multi_query`, `retrieval_batches`, `answer_confidence`, `answer_valid`, `llm_used`, `llm_model`, `conversation_context_used`, `optimization_21` integration flags |

Empty retrieval returns `answer: "I don't know based on the available knowledge base."`, `confidence: 0.0`,
`domain: "unknown"`, `meta.status: "no_answer"`.

---

## 14. Testing and benchmarks

```bash
python -m pytest -q          # 26 test modules under backend/ — run before changing deployment files; keep green
```

Micro-benchmarks (`backend/benchmarks/`) isolate the effect of each optimisation, e.g.:

```bash
python -m backend.benchmarks.benchmark_retrieval_27       # Flat vs HNSW
python -m backend.benchmarks.benchmark_ms_arc_28          # adaptive retrieval depth
python -m backend.benchmarks.benchmark_chunking_40        # chunk-size / overlap experiment
python -m backend.evaluation.fusion_benchmark             # Min-Max vs RRF (Recall/Precision/HitRate/MRR @3)
python -m backend.evaluation.fusion_ablation_report       # summary of fusion_benchmark_results.json
python -m backend.three_way_diagnostic                    # Fixed-K vs Complexity-Adaptive vs MS-ARC
python check_faiss_id_alignment.py                        # FAISS row index ↔ dataset id sanity check
```

**Evaluation protocol (for the paper):** identical corpus, queries and ground truth for every method; ground truth
is verified **manually** (`backend/evaluation/build_fusion_dataset.py` only displays candidates so the benchmark
never becomes circular); metrics are Recall@K, Precision@K, Hit-rate, MRR@K, average retrieval depth and latency.
FAISS returns **row indices** — `data[1885]` means row 1885, not `id == 1885` (the dataset `id` is not globally
unique).

