# PolicyGuard

[![CI](https://github.com/parthivbhesaniya/policy_guard/actions/workflows/ci.yml/badge.svg)](https://github.com/parthivbhesaniya/policy_guard/actions/workflows/ci.yml)

**An agentic, self-correcting, evaluated RAG system for enterprise HR policy Q&A.**

🔗 **Live demo:** [policyguard-parthiv.streamlit.app](https://policyguard-parthiv.streamlit.app)

PolicyGuard answers employee questions strictly from a company's internal policy documents - with forced, verified citations, an LLM-driven hallucination check that retries itself before
answering, human-in-the-loop escalation with real checkpointing for anything it can't verify,
and a golden-dataset evaluation harness to measure all of it. It's built to demonstrate
production RAG engineering, not a demo notebook: every retrieval, generation, and safety
mechanism is designed, tested, and wired the way it would need to be for a real internal tool
handling real policy questions.

---

## Contents

- [PolicyGuard](#policyguard)
  - [Contents](#contents)
  - [Why this project exists](#why-this-project-exists)
  - [Architecture](#architecture)
  - [Key features](#key-features)
  - [Tech stack — why each piece is there](#tech-stack--why-each-piece-is-there)
  - [Project structure](#project-structure)
  - [Getting started](#getting-started)
  - [Quick Command Reference](#quick-command-reference)
  - [PDF support](#pdf-support)
  - [Managing policy documents](#managing-policy-documents)
  - [Usage](#usage)
  - [API \& UI](#api--ui)
  - [Deploying on Streamlit Community Cloud](#deploying-on-streamlit-community-cloud)
  - [Containerization (Docker)](#containerization-docker)
  - [Testing](#testing)
  - [Observability (LangSmith)](#observability-langsmith)
  - [Evaluation \& results](#evaluation--results)
  - [Engineering highlights](#engineering-highlights)

---

## Why this project exists

Internal HR/IT questions ("how much sick leave do I get", "how often do I rotate my
password") are repetitive, but wrong answers on policy topics carry real compliance risk. A
generic chatbot that fabricates an answer when it's unsure is worse than useless here — it has
to know when it doesn't know, prove every claim against a real source, and hand off to a human
rather than guess. PolicyGuard is built around that constraint end-to-end: retrieval, grading,
generation, verification, and escalation all exist specifically to make wrong answers hard to
produce and easy to catch.

## Architecture

![PolicyGuard StateGraph Architecture](./graph_structure.png)

```
data/policies/*.md, *.pdf          data/eval/golden_dataset.json
       │                                       │
       ▼                                       ▼
┌─────────────────────┐              ┌──────────────────────┐
│  Ingestion Pipeline  │              │  Evaluation Harness   │
│  hierarchical chunk  │              │  4 metrics, LangSmith │
│  + table preservation│              │  experiment tracking  │
└──────────┬───────────┘              └───────────▲──────────┘
           ▼                                       │
┌───────────────────────────┐                      │
│   Chroma Vector Store      │                      │
│   parents + children       │◄────────┐            │
└──────────┬─────────────────┘         │            │
           │                    ┌──────┴───────┐    │
           ▼                    │  BM25 Index   │    │
┌────────────────────────────────────────────────────┴───────────────────┐
│                 LangGraph Orchestrator (StateGraph)                    │
│                                                                        │
│  rewrite_query → classify_intent                                       │
│       ├── (greeting) ─────────────► handle_greeting ────────────────┐  │
│       ├── (off-topic) ────────────► handle_out_of_scope ────────────┼─┐│
│       ├── (ambiguous query) ──────► handle_clarification ───────────┼─┼┤
│       └── (policy query) ─────────► retrieve (dense + BM25 → rerank)│ │ │
│                 → grade_documents ──(nothing relevant)──► cannot_ans│ │ │
│                 │ (relevant docs found)                             │ │ │
│                 ▼                                                   │ │ │
│              generate (forced citations) → verify_answer            │ │ │
│                 │              │                                    │ │ │
│                 │        (not grounded, retries left) ─► generate │ │ │
│                 │              │                                    │ │ │
│                 │        (retries exhausted) ─► escalate_to_human   │ │ │
│                 │                                   (interrupt +    │ │ │
│                 │                                    SQLite state)  │ │ │
│                 ▼                                                   ▼ ▼ ▼
│             grounded answer                                         END  
└────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
            answer + citations + audit trail
```

Every arrow above is a real, tested code path — not aspirational. See
[Build status / roadmap](#build-status--roadmap) for what's implemented vs. still planned.

## Key features

- **Hierarchical chunking with Markdown table preservation & PDF parent-child windowing.** Markdown policy docs are split by heading level (`##` parent sections, `###` child subsections) with table-block preservation (`| ... |`) so markdown table headers and rows are never severed across chunk boundaries. PDF docs are chunked by their recovered heading structure — one parent chunk per policy topic (e.g. `Leave Rules (Assistant Level Staff) › Casual Leave`) as LLM context, linked to ~600-char child retrieval chunks via `parent_id` — instead of fixed windows that start mid-word and mix several topics.
- **Smart Intent Guardrails & Pre-Retrieval Routing.** Fast intent classification checks incoming queries before vector search runs, routing non-policy questions away from retrieval:
  - *Greetings*: Inputs like `"hi"` or `"hello"` receive instant friendly welcome responses. A
    self-introduction (`"my name is Harshil"`, `"I'm Alex"`) is recognized as a greeting too and
    gets a personalized `"Hi Harshil!"` reply instead of being mistaken for a policy question.
  - *Out-of-Scope*: General trivia, math, or coding queries receive polite guidance explaining PolicyGuard's policy domain.
  - *Ambiguity Clarification*: Overly broad questions (*"tell me the policy"*) prompt the user to specify their policy topic area (Leave, WFH, IT, Expenses).
- **Hybrid retrieval, not just vector similarity.** Dense embedding search (Chroma) and BM25
  keyword search run over the same corpus and are fused with Reciprocal Rank Fusion, so an exact
  keyword match doesn't lose to a semantically-similar-but-wrong chunk. Fused candidates are then
  reranked by a cross-encoder-style hosted reranker (Cohere) before generation ever sees them.
- **Corrective RAG grading.** An LLM call filters retrieved sections down to the ones actually
  relevant to the question before generation runs — if nothing survives grading, the graph
  answers "I don't have enough information" instead of generating from irrelevant context.
- **Forced, *validated* citations.** Every generated claim must carry an inline
  `[source: doc_id, section]` tag, and a citation isn't trusted just because the model wrote it —
  it's checked against the actual retrieved context, and a citation pointing at content that was
  never retrieved is treated as a grounding failure, not a cosmetic one.
- **Conversational follow-ups, without letting the model answer from memory.** A caller (CLI,
  API, or UI) can pass prior Q&A turns alongside a new question, and `rewrite_query` uses them
  to resolve references like "does *this* apply to interns" or "what about *that*" into a
  standalone search query. The resolved query — not the raw, ambiguous one — is what grading and
  generation see, so retrieval and citation-checking work correctly on a follow-up. Conversation
  history is deliberately *never* shown to `generate` itself: every claim in the final answer
  still has to come from retrieved, graded policy excerpts, not from something said earlier in
  the chat.
- **A hallucination check that can act on what it finds.** A second LLM pass verifies the
  generated answer is fully supported by the retrieved context. If not, the graph loops back to
  `generate` (bounded retries) before giving up — it doesn't just log a "low confidence" score
  and ship the answer anyway.
- **Real human-in-the-loop, not a TODO comment.** When retries are exhausted, the graph pauses
  mid-execution via LangGraph's `interrupt()` and persists full state to a SQLite checkpoint
  under a `thread_id`. A *separate* CLI invocation, in a separate process, resumes it later with
  an approve/edit/reject decision — proving the "resumable across sessions" requirement rather
  than faking it with an in-memory `input()` prompt.
- **A real evaluation harness, not a vibe check.** A 49-example golden Q&A dataset (covering
  26 distinct sections of the ingested policy document plus 12 deliberately unanswerable
  questions) is scored on four metrics — two deterministic, two LLM-judged — locally or as a
  tracked LangSmith experiment. On top of that, a **DeepEval suite** scores retrieval,
  generation, and the full pipeline as three separate stages (Contextual Recall/Precision,
  Faithfulness, Answer Relevancy, Contextual Relevancy), judged by a larger model than the one
  that generates the answers — see [Evaluation & results](#evaluation--results).
- **Fully dependency-injected.** The LLM, retriever, reranker, and checkpointer are all
  swappable at the function-call boundary. The entire 121-test suite runs against fakes/temp
  local stores in ~20 seconds — no live API calls, no network flakiness, no API cost to run CI.

## Tech stack — why each piece is there

| Layer | Technology | What it's doing here |
| --- | --- | --- |
| LLM inference | **Groq** (`openai/gpt-oss-20b` by default, configurable via `GROQ_MODEL`) via `langchain-groq` | Fast, free-tier-friendly inference for every LLM call in the graph — rewriting, grading, generation, verification, and evaluation judging all go through one interchangeable `BaseChatModel`. |
| Orchestration | **LangGraph** `StateGraph` | The centerpiece: models the agent as an explicit graph with conditional routing (`grade_documents` → generate or bail) and a real cycle (`verify_answer` → retry `generate`), not a linear chain pretending to be an agent. |
| Vector search | **ChromaDB** (persistent, local embeddings) | Stores parent/child chunk collections and runs dense similarity search — no external embedding API required. |
| PDF ingestion | **pypdf** | Pure-Python text extraction for PDF policy docs (no compiled/native deps) — no torch-style wheel-availability risk. |
| Keyword search | **rank-bm25** | Classic sparse retrieval over the same child chunks, built in-memory from the Chroma collection — catches exact-term queries dense embeddings can miss. |
| Fusion | Reciprocal Rank Fusion (hand-rolled) | Combines the dense and BM25 rankings into one candidate pool without needing either signal to dominate by construction. |
| Reranking | **Cohere** Rerank API | Cross-encoder-quality reranking of the fused candidates. (A local `sentence-transformers` cross-encoder was the first choice — see [Engineering highlights](#engineering-highlights) for why that had to change.) |
| Prompts / messages | **langchain-core** | `SystemMessage`/`HumanMessage` primitives and `BaseChatModel` typing, used directly rather than through a heavier chain abstraction that the graph doesn't need. |
| Human-in-the-loop | **LangGraph** `interrupt()` / `Command(resume=...)` | Pauses graph execution mid-node and resumes it later from an entirely different process invocation. |
| Persistence | **langgraph-checkpoint-sqlite** | Durable, on-disk checkpointing of full graph state per `thread_id`, so paused conversations survive process restarts. |
| Evaluation & tracing | **LangSmith** | Traces every graph run (each node's prompt, output, tokens, latency) from the UI, API, CLI, and evals into one project, labelled by source and outcome, with user 👍/👎 feedback — plus optional experiment tracking for `run_eval`. |
| RAG evaluation | **DeepEval** (judge: Groq `openai/gpt-oss-120b`) | Component-level (retrieval, generation) and end-to-end (RAG triad) LLM-judged metrics as pytest tests. A custom `DeepEvalBaseLLM` routes the judge through Groq with structured JSON output, and deliberately uses a bigger model than the generator so the judge isn't grading its own answers. |
| Config | **python-dotenv** | Loads API keys from a gitignored `.env` — nothing secret is hardcoded or committed. |
| API | **FastAPI** | Thin HTTP wrapper around the same compiled graph the CLI uses — `/ask` and `/resolve`, including the interrupt/resume flow, over a stable JSON contract instead of stdin/stdout. |
| UI | **Streamlit** | Minimal chat UI on top of the API — question in, cited answer out, with approve/edit/reject controls when a thread pauses for human review. |
| Testing | **pytest** | 142 tests across every layer, built almost entirely on fakes (`FakeLLM`, `FakeCohereClient`) and real-but-temporary Chroma stores (`tmp_path`) instead of mocking/patching internals. |
| Containerization | **Docker** / Compose | An `api` + `ui` service pair sharing one image, plus a one-off `ingest` profile — persists `chroma_db` and `checkpoints.sqlite` to the host and reuses the host's Chroma embedding-model cache instead of re-downloading it per container. |

## Project structure

```
policyguard/
├── data/
│   ├── policies/                 # Policy docs to ingest (Markdown + YAML front matter, or PDF)
│   └── eval/
│       ├── golden_dataset.json     # 49-example golden Q&A set (question, answer, source, answerable)
│       └── generation_golden.json  # ideal context per answerable question (verbatim Chroma chunks)
├── src/policyguard/
│   ├── ingestion/                # Loading, chunking, Chroma vector store
│   │   ├── loader.py             #   Markdown: parses YAML front matter + body
│   │   ├── pdf_loader.py         #   PDF: extracts text, reads sidecar .yaml (or guesses defaults)
│   │   ├── pdf_structure.py      #   PDF: recovers parts / sections / topics / clauses from extracted text
│   │   ├── chunker.py            #   Markdown: ## → parent, ### → child + table preservation. PDF: one parent per topic
│   │   ├── vectorstore.py        #   Chroma-backed store: query, get-by-id, get-all, delete
│   │   └── ingest.py             #   CLI: ingest docs (.md or .pdf) / run a raw retrieval query
│   ├── generation/                # Linear "baseline" generate-with-citations chain
│   │   ├── chain.py               #   retrieve → prompt → LLM → validate citations
│   │   ├── prompts.py             #   forced-citation system prompt
│   │   ├── citations.py           #   citation parsing + validation
│   │   └── ask.py                 #   CLI
│   ├── retrieval/                  # Hybrid search + reranking
│   │   ├── bm25.py                 #   BM25 index over child chunks
│   │   ├── hybrid.py               #   dense + BM25 fusion via RRF
│   │   └── reranker.py             #   Cohere rerank wrapper
│   ├── orchestration/               # The LangGraph StateGraph
│   │   ├── state.py                 #   GraphState schema
│   │   ├── nodes.py                 #   nodes, intent classifiers (greeting/out-of-scope/ambiguity), routing
│   │   ├── graph.py                 #   graph wiring / compilation
│   │   ├── ask.py                   #   CLI: ask a question through the graph
│   │   └── resolve.py               #   CLI: resume a paused/escalated conversation
│   ├── evaluation/                   # Golden dataset + evaluators
│   │   ├── dataset.py                 #   golden example loader
│   │   ├── evaluators.py              #   recall@k, citation_accuracy, faithfulness, answer_relevance
│   │   ├── run_eval.py                #   CLI: local scorecard or LangSmith experiment
│   │   └── deepeval_judge.py          #   GroqJudge: DeepEval judge LLM on Groq
│   ├── observability.py              # LangSmith trace labels, outcome + 👍/👎 feedback
│   ├── api/                          # FastAPI wrapper around the compiled graph
│   │   ├── app.py                     #   /ask, /resolve, /health
│   │   └── schemas.py                 #   request/response pydantic models
│   └── ui/                           # Minimal Streamlit UI
│       └── app.py                     #   chat UI calling the API, incl. review approve/edit/reject & clear chat
├── evals/                              # DeepEval suite (live, billed LLM calls -- not run by `pytest`)
│   ├── conftest.py                     #   shared fixtures: judge, store, retriever, reranker, generator
│   ├── test_retrieval.py               #   stage 1: Contextual Recall + Precision
│   ├── test_generation.py              #   stage 2: Faithfulness + Answer Relevancy (ideal context)
│   └── test_pipeline.py                #   stage 3: RAG triad over the full LangGraph app
└── tests/                              # 142 tests, one file per module above
```

## Getting started

```bash
git clone <repo-url> && cd policyguard
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"   # add `,ui` only if you want the Streamlit UI; `,eval` for the DeepEval suite

cp .env.example .env
# then edit .env and add your API keys (see below)
```

`.env` variables:

| Variable | Required for | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | Everything past ingestion | Free tier at [console.groq.com](https://console.groq.com/keys) |
| `GROQ_MODEL` | — | Defaults to `openai/gpt-oss-20b` |
| `COHERE_API_KEY` | Reranking (or pass `--no-rerank`) | Free tier at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) |
| `LANGSMITH_API_KEY` | Tracing, `run_eval --langsmith` | Optional — with `LANGSMITH_TRACING=true` every run is traced (see [Observability](#observability-langsmith)) |
| `POLICYGUARD_ENV` | — | Labels traces by deployment, e.g. `demo` on Streamlit Cloud; defaults to `local` |
| `DEEPEVAL_JUDGE_MODEL` | — | DeepEval judge model; defaults to `openai/gpt-oss-120b` |
| `EVAL_INCLUDE_REASON` | — | Set to `1` to have DeepEval write the judge's reasoning per metric (off by default to save tokens) |

Drop your own policy docs (Markdown or PDF) into `data/policies/`, then ingest them into a local
Chroma store:

```bash
python -m policyguard.ingestion.ingest --input data/policies --persist-dir ./chroma_db
```

## Quick Command Reference

Here is a quick summary of essential commands for running, managing, and clearing PolicyGuard:

### 1. Clear / Reset Vector DB (Clean Fresh Start)
To wipe the existing Chroma DB vector store and clear all past chunk embeddings:
```bash
python -c "import shutil; shutil.rmtree('./chroma_db', ignore_errors=True); print('Chroma DB vector store cleared!')"
```

### 2. Ingest Policy Documents
To ingest Markdown (`.md`) or PDF (`.pdf`) policy files from `data/policies/` into Chroma DB:
```bash
python -m policyguard.ingestion.ingest --input data/policies --persist-dir ./chroma_db
```

### 3. Check Vector DB Chunk Count
To verify how many parent and child chunks are currently indexed in Chroma DB:
```bash
python -c "from pathlib import Path; from policyguard.ingestion.vectorstore import PolicyVectorStore; store = PolicyVectorStore(Path('./chroma_db')); print('Parents:', store._parents.count(), 'Children:', store._children.count())"
```

### 4. Run Web Application (API + UI)
- **Step 1: Start FastAPI Service (Backend):**
  ```bash
  uvicorn policyguard.api.app:app --reload
  ```
- **Step 2: Start Streamlit Interface (Frontend - in a new terminal tab):**
  ```bash
  streamlit run src/policyguard/ui/app.py
  ```

### 5. Run CLI Interactive Mode
```bash
python -m policyguard.orchestration.ask
```

### 6. Run Test Suite
```bash
pytest
```

## PDF support

`data/policies/` (or any `--input` directory) can mix `.md` and `.pdf` policy docs. A PDF has
no YAML front matter and no `##`/`###` heading structure to hierarchically chunk by, so:

- **Metadata** comes from an optional sidecar YAML file with the same name: `finance_policy.pdf` +
  `finance_policy.yaml`, with the same four fields as Markdown front matter:
  ```yaml
  doc_id: finance-expense-policy
  department: Finance
  effective_date: 2026-01-01
  version: 1.0
  ```
  If a sidecar exists, all four fields are required. If there's no sidecar at all, a PDF can
  still be dropped in with zero setup — metadata is auto-generated from the filename (`doc_id`
  slugified from the stem, `department` guessed from keywords like "hr"/"security"/"finance",
  `effective_date` defaulted to today, `version` to `"1.0"`), with a printed warning so it's
  obvious the values were guessed rather than authored.
- **Chunking** recovers the PDF's heading structure from its text
  ([pdf_structure.py](src/policyguard/ingestion/pdf_structure.py)): parts (`PART V` + title),
  numbered sections (`7. Leave Rules`), lettered topics (`(b) Work From Home (WFH).`) and titled
  clauses (`(ii) Casual Leave.`), after stripping running page headers, page numbers, the table of
  contents, and Cyrillic/Greek look-alike letters from extraction. Each topic becomes one parent
  chunk (generation context, up to ~2,000 chars, longer topics split into `(cont. N)` parts) with
  ~600-char children for retrieval, and is cited by its last two heading levels:
  `[source: hr-policy-dec-2025, Leave Rules (Assistant Level Staff) › Casual Leave]`. A PDF with no
  recoverable headings falls back to paragraph-aware fixed-size windows cited as `Part N`.

Text extraction uses `pypdf` (pure Python, no compiled/native dependencies). Ingest exactly the
same way — the CLI auto-detects file type by extension:

```bash
python -m policyguard.ingestion.ingest --input data/policies --persist-dir ./chroma_db
```

## Managing policy documents

Ingestion always `upsert`s — it can add or update, but it never removes. Whether re-running it
after a change is enough, or you need to clean up manually, depends on what changed:

| You changed... | What to do | Why |
| --- | --- | --- |
| Content within an existing section (same file, same `doc_id`, same heading) | Just re-run ingestion | Chunk IDs are deterministic hashes of `doc_id` + section title, so the same chunk gets overwritten in place. Safe, idempotent, no cleanup needed. |
| A section's heading text, or removed a section | Re-run ingestion, **then** manually remove the orphaned chunk | The old heading produces a different chunk ID than the new one, so the old chunk is never overwritten — it just sits there, findable by retrieval, pointing at content that no longer exists in the source file. |
| A `doc_id` (front matter, sidecar YAML, or a PDF filename with no sidecar — its `doc_id` is guessed from the filename) | Delete the **old** `doc_id` before or after ingesting under the new one | Same root cause as above, at the whole-document level: the store now has two `doc_id`s for what you consider one logical document. |
| Removed a policy file entirely | Delete its `doc_id` | Ingestion only ever processes files it currently finds in `--input` — it has no notion of "this used to exist and doesn't anymore." |

There's no CLI command for cleanup yet — do it directly:

```bash
python -c "
from pathlib import Path
from policyguard.ingestion.vectorstore import PolicyVectorStore

store = PolicyVectorStore(Path('./chroma_db'))
store.delete_document('the-old-doc-id')
"
```

To see what's actually in the store (useful before deleting anything, or after a re-ingest to
sanity-check what landed):

```bash
python -c "
from pathlib import Path
from collections import Counter
from policyguard.ingestion.vectorstore import PolicyVectorStore

store = PolicyVectorStore(Path('./chroma_db'))
_, _, metadatas = store.get_all_children()
print(Counter(m['doc_id'] for m in metadatas))
"
```

This gap (upsert-only, no automatic pruning of stale `doc_id`s) is a known limitation — see
[Engineering highlights](#engineering-highlights) for two real incidents it caused.

## Usage

**Raw retrieval only** (no LLM, sanity-checks ingestion):
```bash
python -m policyguard.ingestion.ingest --query "how many carryover leave days am I allowed"
```

**Simple baseline chain** (retrieve → generate with citations, no grading/self-correction):
```bash
python -m policyguard.generation.ask "how many carryover leave days am I allowed"
```

**Full agentic pipeline** (hybrid retrieval + reranking + grading + hallucination check):
```bash
python -m policyguard.orchestration.ask "how many carryover leave days am I allowed"
python -m policyguard.orchestration.ask "..." --no-rerank   # skip Cohere, no key needed
python -m policyguard.orchestration.ask   # omit the question for interactive mode
```
If the answer can't be verified as grounded after retries, this prints a `thread id` and pauses
instead of returning an answer. Interactive mode remembers the session's prior questions and
answers, so a follow-up like "does this apply to interns" resolves against what was already
asked (see [Key features](#key-features)).

**Resolve a paused/escalated conversation** (a separate process, proving real resumability):
```bash
python -m policyguard.orchestration.resolve --thread-id <id> --action approve
python -m policyguard.orchestration.resolve --thread-id <id> --action edit --answer "..."
python -m policyguard.orchestration.resolve --thread-id <id> --action reject
```

**Run the evaluation suite:**
```bash
python -m policyguard.evaluation.run_eval                # local scorecard
python -m policyguard.evaluation.run_eval --langsmith     # + tracked LangSmith experiment
```

**Run the DeepEval suite** (needs `pip install -e ".[eval]"` and an ingested `chroma_db`; each
stage runs on its own):
```bash
deepeval test run evals/test_retrieval.py    # stage 1: retrieval
deepeval test run evals/test_generation.py   # stage 2: generation
deepeval test run evals/test_pipeline.py     # stage 3: full pipeline
deepeval test run evals/test_pipeline.py -k "office-hours-01 or tor-approval-01"   # a subset
EVAL_INCLUDE_REASON=1 deepeval test run evals/test_generation.py                   # with judge reasons
```

## API & UI

A FastAPI service exposes the same compiled LangGraph app the CLI uses, over HTTP instead of
stdin/stdout — same hybrid retrieval, grading, hallucination check, and human-in-the-loop
escalation, just a different front door. Setup (Chroma connection, BM25 index, LLM, checkpointer,
graph compilation) happens once at process startup and is shared across every request.

```bash
uvicorn policyguard.api.app:app --reload
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/ask` | POST | `{"question": "...", "history": [...]}` → `{"status": "answered" \| "cannot_answer" \| "needs_review", "answer": ..., "citations": [...], ...}` |
| `/resolve` | POST | `{"thread_id": ..., "action": "approve" \| "edit" \| "reject", "answer": ...}` — resumes a thread paused by `/ask` |
| `/health` | GET | Liveness check |

A `needs_review` response means the answer couldn't be verified as grounded after retries; it's
paused (checkpointed, exactly like the CLI's escalation) until a `/resolve` call decides its fate
— from the same request, a later one, or an entirely different client.

`history` is optional and defaults to `[]` — pass prior turns
(`[{"question": ..., "answer": ...}, ...]`, oldest first) so a follow-up question like "does this
apply to interns" can be resolved against what was already asked. The API is otherwise stateless
across calls, so accumulating and resending history is the caller's responsibility — the
Streamlit UI does this automatically from its own chat history.

A minimal Streamlit UI sits on top of the API — a chat box in, a cited answer out, with
approve/edit/reject buttons that appear automatically when a thread pauses for review:

```bash
# with the API already running (see above)
streamlit run src/policyguard/ui/app.py
```

## Deploying on Streamlit Community Cloud

[ui/app.py](src/policyguard/ui/app.py) is self-contained: it builds and runs the LangGraph
orchestrator directly in-process (`build_graph()` + `.invoke()`/`.stream()`), rather than calling
the FastAPI service over HTTP. This is what makes it deployable on Streamlit Community Cloud as a
single app, with nothing else to host.

1. Push this repo to GitHub, then create a new app at [share.streamlit.io](https://share.streamlit.io)
   pointing at it.
2. **Main file path:** `src/policyguard/ui/app.py`
3. **App settings → Secrets**, add at minimum:
   ```toml
   GROQ_API_KEY = "..."
   GROQ_MODEL = "..."       # must be a model your Groq key actually has access to -- see below
   COHERE_API_KEY = "..."   # optional; reranking is skipped automatically if omitted
   # optional -- LangSmith tracing + 👍/👎 feedback (see Observability below)
   LANGSMITH_TRACING = "true"
   LANGSMITH_API_KEY = "..."
   LANGSMITH_PROJECT = "..."
   POLICYGUARD_ENV = "demo"
   ```
4. Deploy. The repo's [requirements.txt](requirements.txt) (`-e .[ui]`, installing this package
   plus its `ui` extra from `pyproject.toml`) and [packages.txt](packages.txt) (apt libs the PDF
   OCR fallback needs) are picked up automatically by Streamlit Cloud's build.

A few things specific to this deployment path, not the local/Docker one:

- **No `POLICYGUARD_API_URL` to set** — there's no second service. The FastAPI app
  ([api/app.py](src/policyguard/api/app.py)) still exists and still works for local/Docker use,
  it's just not part of this deployment.
- **Ephemeral filesystem.** Streamlit Cloud's disk resets on every restart/redeploy, so
  `chroma_db/` doesn't persist between cold starts. `ui/app.py` handles this itself: on first
  load it checks whether the vector store is empty and, if so, ingests everything in
  `data/policies/` before serving any questions (cached for the app's lifetime via
  `st.cache_resource`, so it only happens once per restart, not once per user). The same goes for
  the human-review checkpoint state — it uses an in-memory `InMemorySaver` instead of the local
  `SqliteSaver`, since a durable file wouldn't survive a restart here either. A paused
  `needs_review` thread only survives as long as the app process stays up.
- **Verify your `GROQ_MODEL` before deploying.** Groq's available model list is account/key
  specific and changes over time — even this README's documented default (`openai/gpt-oss-20b`)
  may not exist for your key. Check what your key can actually use:
  ```bash
  curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY" | jq -r '.data[].id'
  ```
  and also check the output-tokens-per-minute limit on whatever model you pick — a low-tier
  model's OTPM cap can be too small for a full policy answer with citations, which surfaces as an
  HTTP 429 from Groq.

## Containerization (Docker)

The whole stack (API + UI) also runs without a local Python environment at all — just Docker.

```bash
cp .env.example .env   # if you haven't already; the containers read the same file
docker compose up -d api ui
```

- **API** → http://localhost:8000 (`/health`, `/ask`, `/resolve`, `/docs`)
- **UI** → http://localhost:8501 (talks to the `api` service by its Compose network name, via
  the `POLICYGUARD_API_URL` env var — no manual URL entry needed)

Re-run ingestion inside a container (useful if you don't want `pip install`ed locally at all):

```bash
docker compose --profile tools run --rm ingest
```

```bash
docker compose down            # stop everything
docker compose logs -f api     # tail a service's logs
```

A few things about how this is wired, worth knowing before you change it:

- **[Dockerfile](./Dockerfile)** is a single `python:3.11-slim` stage with no compiler toolchain
  — every dependency here (`chromadb`, `onnxruntime`, `pypdfium2`, `opencv-python` via
  `rapidocr-onnxruntime`) ships a prebuilt manylinux wheel, so `build-essential` isn't needed and
  skipping it keeps both the image and the build itself lighter. The only apt packages installed
  are `libgl1`/`libglib2.0-0`, which `opencv-python` needs at runtime.
- **[docker-compose.yml](./docker-compose.yml)** defines `api`, `ui`, and a one-off `ingest`
  profile, all built from the same image. `chroma_db/` and `checkpoints.sqlite` are bind-mounted
  to the host so the vector store and LangGraph checkpoints persist across `docker compose down`
  the same way they would running locally.
- It also bind-mounts `~/.cache/chroma` into the containers. Chroma's default embedding function
  lazily downloads a ~79 MB ONNX model (`all-MiniLM-L6-v2`) on first use; without this mount, a
  fresh container re-downloads it on every rebuild, which is slow on a constrained connection.
  Reusing the host's copy means containers start warm.
- The `api` service has a Compose healthcheck against `/health`; `ui` declares
  `depends_on: api: condition: service_healthy`, so the UI never starts pointing at a backend
  that isn't ready yet.

## Testing

```bash
pip install -e ".[dev]"
ruff check .   # lint: pyflakes, serious pycodestyle errors, import order (rules in pyproject.toml)
pytest
```

Both run on every push via GitHub Actions ([ci.yml](.github/workflows/ci.yml)); the badge at the
top of this README shows the latest result.

142 tests, ~20 seconds, zero live API calls (LangSmith tracing is forced off in `tests/conftest.py`) (`pytest` only collects `tests/`; the DeepEval
suite in `evals/` makes real LLM calls and runs separately — see
[Evaluation & results](#evaluation--results)):

| File | Tests | Covers |
| --- | --- | --- |
| `test_chunker.py` | 8 | Hierarchical chunking, parent/child linking, metadata propagation |
| `test_pdf_ingestion.py` | 21 | Flat window chunking, overlap, sidecar YAML metadata validation, guessed-default fallback |
| `test_chain.py` | 5 | Context-block deduping, prompt construction |
| `test_citations.py` | 5 | Citation parsing (incl. fullwidth `【source: …】` brackets) + validation |
| `test_retrieval.py` | 9 | BM25 exact-match, RRF fusion, Cohere reranker (via fake client) |
| `test_orchestration.py` | 37 | Every node, every routing decision (incl. self-introduction greetings), conversation-history resolution, full-graph integration incl. interrupt/resume, via a `FakeLLM` |
| `test_evaluation.py` | 16 | Dataset integrity, both programmatic metrics, both LLM-judge metrics (via fake LLM) |
| `test_vectorstore.py` | 2 | `delete_document` removes only the targeted doc's chunks, no-ops for an unknown doc id |
| `test_api_streaming.py` | 1 | Streaming `/ask` response shape |
| `test_observability.py` | 17 | Trace labels, outcome classification, feedback sending (via a fake LangSmith client), no-op when tracing is off |
| `test_pdf_structure.py` | 12 | PDF heading recovery: header/page-number/TOC cleanup, look-alike letters, topic vs clause vs list-item headings, labelled chunks, word-boundary splitting |
| `test_ui_errors.py` | 9 | Groq/Cohere errors mapped to friendly UI messages; raw error logged, never shown |

The orchestration tests are the ones worth highlighting: they run the **actual compiled
LangGraph app** — including the interrupt/checkpoint/resume cycle against a real
`InMemorySaver` — with only the LLM swapped for a scripted fake, so the graph wiring itself is
verified, not just each node in isolation.

## Observability (LangSmith)

Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` (in `.env` locally, in
the app's Secrets on Streamlit Cloud) and every question becomes a LangSmith trace: one
`PolicyGuard` run with a child per graph node — `rewrite_query → retrieve → grade_documents →
generate → verify_answer` (plus retries) — showing each prompt, model output, token count, cost,
and latency. When a user reports a wrong answer, the trace shows whether retrieval, grading, or
generation went wrong. Tracing itself is automatic; [observability.py](src/policyguard/observability.py)
adds what a raw trace can't tell you:

| Signal | Recorded as | Values |
| --- | --- | --- |
| Where the run came from | trace tags + metadata | `entrypoint`: `ui` / `api` / `cli` / `eval`; `environment`: `POLICYGUARD_ENV` (e.g. `demo` vs `local`) |
| How it ended | feedback `outcome` | `answered`, `cannot_answer`, `escalated`, `human_reviewed`, `greeting`, `out_of_scope`, `clarification` |
| Self-correction | feedback `retries` | generation attempts beyond the first (verify loop rejected an answer) |
| Citation problems | feedback `invalid_citations` | citations pointing at sections that weren't retrieved |
| What users think | feedback `user_score` | 👍 = 1 / 👎 = 0, from the buttons under each answer in the Streamlit UI |

Everything goes into **one LangSmith project**; filter by the `entrypoint` / `environment` tags
to separate live demo traffic from local runs and eval runs (the DeepEval suite tags its traces
`entrypoint:eval`). The Monitoring tab then charts latency, tokens, and errors out of the box,
and the feedback keys above give escalation, cannot-answer, retry, and 👍/👎 rates over time.

![LangSmith trace of a live demo question: the node waterfall, with the user's 👍 and the outcome, retries, and invalid_citations feedback](./Lang_Smith_Dashboard/LS_Dashboard.jpeg)

*A live demo question in LangSmith: each graph node's timing on the left (2.26 s, ~3.9K tokens,
$0.0005 in total), and on the right the user's 👍 (`user_score 1.00`) next to the automatic
`outcome`, `retries`, and `invalid_citations` feedback.*

Observability never gets in the way: with tracing off every helper is a no-op (and the 👍/👎
buttons are hidden), feedback is sent from a background thread so answers aren't delayed, and a
LangSmith outage is logged rather than raised. The UI tells users their questions are logged
whenever tracing is on.

## Evaluation & results

### DeepEval suite: retrieval, generation, and the full pipeline

The DeepEval suite in [evals/](evals/) measures each part of the RAG system on its own, then the
whole thing end to end, so a bad score points at a specific component instead of "the answer
was wrong somewhere". All three stages use the 37 answerable golden questions (the 12
deliberately unanswerable ones have no correct context to score against), a pass mark of
**0.7 for every metric except Contextual Relevancy** (report-only, explained below), and
`openai/gpt-oss-120b` as the judge — a different, larger model than the `openai/gpt-oss-20b`
generator. Results below are from 2026-10-02, on the structure-aware topic chunks.

| Stage | What runs | Metrics | Result (37 questions) |
| --- | --- | --- | --- |
| 1. Retrieval | The `retrieve` node alone (hybrid search → Cohere rerank → top 4 parent chunks), on the raw question | Contextual Recall, Contextual Precision | **36 / 37 pass** (recall 1.0 on all 37) |
| 2. Generation | The `generate` node alone, fed the *ideal* context from [generation_golden.json](data/eval/generation_golden.json) so retrieval mistakes can't leak in | Faithfulness, Answer Relevancy | **35 / 37 pass** (both failures are judge errors, below) |
| 3. Pipeline | The full LangGraph app: query rewrite → retrieve → grade → generate → verify/retry | RAG triad: Answer Relevancy, Faithfulness (+ Contextual Relevancy, report-only) | **37 / 37 pass** — Faithfulness ≥ 0.8 and Answer Relevancy ≥ 0.75 on every question (1.0 on 36) |

What the failures actually mean:

- **Retrieval: `contract-termination-notice-01` (Contextual Precision 0.5)** — a real ranking
  weakness. The question asks about notice "outside of **probation**", so the Probation chunk
  outranks the Contract Agreement chunk that holds the answer (rank 2). Recall is 1.0 and end to
  end it doesn't matter: stage 3 answers it perfectly (Faithfulness and Answer Relevancy 1.0).
- **Generation: two judge false negatives.** Both answers were read by hand and are correct and
  fully grounded (Faithfulness 1.0); the judge scored Answer Relevancy 0.67 on each.
  `external-consultant-empanelment-01` restates the golden answer almost word for word;
  `grievance-committee-01` lists the right members plus one true extra sentence the judge called
  off-question. Which questions trip the judge varies run to run — both runs are documented in
  [test_generation.py](evals/test_generation.py).
- **Pipeline: Contextual Relevancy measures sentences, not chunks** (mean 0.32; ≥ 0.7 on 2 of
  37). It scores the share of the context's *statements* relevant to the question, so a narrow
  question scores low against even a perfect context: "When is a medical certificate required for
  sick leave?" retrieves exactly the four-sentence Sick Leave clause, and the judge marks only the
  certificate rule relevant — not "ten days per year", "cannot be encashed", or "carried over up
  to 20 days" — for 0.25. It's reported for trends but doesn't fail tests; Faithfulness and
  Answer Relevancy are the pass/fail signals.

**Before/after: fixed windows → topic chunks.** The first full run (2026-09-29/10-01) used
~2,000-char fixed windows labelled `Part N`, which started mid-word and mixed several topics
(one window held the end of Casual Leave, all of Sick Leave, and Special Leave). Contextual
Relevancy failed on 36 of 37 questions, which looked like a chunking problem, so the PDF chunker
was rebuilt to follow the document's own headings (see [PDF support](#pdf-support)) and all three
stages re-run:

| | Fixed windows | Topic chunks |
| --- | --- | --- |
| Parent chunks (median size) | 79 (~1,730 chars) | 112 (~640 chars) |
| Citations look like | `Part 34` | `Leave Rules (Assistant Level Staff) › Sick Leave` |
| Context sent to the generator (5-question sample) | ~2,470 chars | ~790 chars (**−68%**) |
| Stage 1 retrieval | 36 / 37 (`tor-approval-01` fails: rank 3–4) | 36 / 37 (`tor-approval-01` now passes; `contract-termination-notice-01` fails) |
| Stage 2 generation | 35 / 37 | 35 / 37 |
| Stage 3 Faithfulness / Answer Relevancy | 1.0 / ≥ 0.75 on all 37 | ≥ 0.8 / ≥ 0.75 on all 37 |
| Stage 3 Contextual Relevancy | 0.06–0.62 (≥ 0.7 on 1) | 0.09–1.0, mean 0.32 (≥ 0.7 on 2) |

The rebuild kept answer quality the same while cutting the context, and so Groq tokens and
latency, by about two-thirds, and made every citation human-readable. It barely moved Contextual
Relevancy, which is how the sentence-level explanation above was found: the metric was never
measuring chunk size. Getting it past 0.7 would need per-question sentence filtering — an extra
LLM call per question spent satisfying the metric rather than improving answers.

A few design choices worth knowing if you run or extend it:

- **Kept out of `pytest`.** Every metric is a live, billed judge call, so the suite lives in
  `evals/` and `pyproject.toml` sets `testpaths = ["tests"]` — a plain `pytest` never spends tokens.
- **Free-tier friendly.** Judge reasoning is off by default (`EVAL_INCLUDE_REASON=1` turns it
  on), and the judge retries Groq rate limits with backoff. Stage 3 stops cleanly when Groq's
  daily token quota runs out: that test is marked "not scored" and the rest are skipped without
  running the pipeline, so re-running the remainder with `-k` loses nothing. On Groq's free tier
  (200K judge tokens/day) the full suite takes a few days; stage 3 alone is ~15K tokens per question.
- **Ideal context is pinned to the vector store.** `generation_golden.json` holds chunks copied
  verbatim from Chroma; stage 2 fails with a clear message if re-ingestion ever changes them.

### Earlier harness: `run_eval` scorecard

Latest run of the full 49-example golden dataset against the real ingested policy document
(`hr-policy-dec-2025`, a 38-page government HR/admin policy PDF), with `llama-3.3-70b-versatile`
(the project's default model at the time of this run — since deprecated by Groq and replaced by
`openai/gpt-oss-20b`; these numbers haven't been re-measured against the new default yet):

| Metric | Score |
| --- | --- |
| recall@k | 0.88 |
| citation_accuracy | 0.61 |
| faithfulness | 1.00 |
| answer_relevance | 0.98 |

Faithfulness is perfect across all 49 questions — no hallucinated claims. citation_accuracy is
the weak spot, and `run_eval`'s per-example failure log points at two distinct, fixable causes
rather than one vague "citations are unreliable":

1. **Citations omitted on correct answers.** Several answers are factually right but ship with
   no `[source: ...]` tag at all, so `citation_accuracy` scores 0 even though nothing is
   factually wrong.
2. **Citation granularity drift.** The model sometimes cites a sub-clause or appendix label
   (`Part 14(b)(i)`, `Part VI (g)`, `Appendix-A, 5`) instead of the exact `Part N` string the
   chunk is actually indexed under, so citation-validation rejects it as an "invalid citation"
   even when it's pointing at essentially the right place.

recall@k misses are concentrated on facts that sit in overlapping/boundary-adjacent chunks (e.g.
a leave entitlement whose full description spans two consecutive `Part N` sections) — retrieval
sometimes surfaces the semantically-similar neighbor chunk instead of the exact one the golden
example was graded against.

## Engineering highlights

A few things worth calling out that came from actually exercising the system against a live
LLM, not just reading the architecture doc:

- **Found and fixed a real citation-accuracy bug.** The LLM would sometimes cite a `###`
  subsection heading it saw *inside* an excerpt's body text instead of the parent section given
  in that excerpt's own `[source: ...]` tag. Caught by the citation-validation layer, but the
  graph wasn't acting on it — a bad citation could still reach the user as long as the answer's
  prose was independently judged "faithful." Fixed at two levels: tightened the generation
  prompt to explicitly disallow it, and wired citation validity into the groundedness check so
  the self-correction loop treats a bad citation as a verification failure, not a footnote.
- **Distinguished evaluator bugs from system bugs.** A first baseline run showed
  faithfulness/relevance failures with no visible reason. Fixing the evaluator to surface the
  judge's actual reasoning (instead of just a score) revealed the *judge* was wrong, not the
  system: it failed closed on answers with zero factual claims (a correct "I don't know" has
  nothing to be unsupported — it's vacuously faithful), and it penalized correctly-declined
  out-of-scope questions using its own world knowledge instead of respecting that this assistant
  is deliberately restricted to the policy corpus. Both judge prompts were corrected accordingly.
- **Hit a real platform constraint and adapted rather than fought it.** The architecture doc's
  suggested local cross-encoder reranker needs `sentence-transformers`/`torch`, which has no
  installable wheel for an Intel/x86_64 Mac on Python 3.13 (recent PyTorch macOS builds are
  Apple-Silicon-only). Rather than forcing it, reranking moved to a hosted API (Cohere) behind
  the same interface, with a `--no-rerank` escape hatch that needs no key at all.
- **Found a second stale-context bug while adding conversational memory.** After wiring
  conversation history into `rewrite_query`, a live follow-up question ("does this apply to
  interns") still failed — retrieval found the right excerpt, but `grade_documents` and
  `generate` were still being prompted with the raw, ambiguous question text, not the
  disambiguated `rewritten_query`. The grader/generator has no access to conversation history by
  design (so answers can't smuggle in unretrieved "facts" from earlier in the chat), so an
  unresolved "this" was unanswerable to them even with the correct excerpt in hand. Fixed by
  routing `rewritten_query` through both nodes instead of `question`.
- **Caught a corpus contamination issue the same follow-up question exposed.** Even after that
  fix, the same live question came back wrong until digging into what was actually retrieved: the
  vector store held chunks from two unrelated organizations' HR policies under different
  `doc_id`s (the real company policy, plus an unrelated government HR manual that had been
  ingested at some point) — Cohere's reranker had no way to know which org "this" referred to, so
  it silently arbitrated between them. `delete_document()` (added earlier for a different
  duplicate) removed the stray document; confirms that data hygiene, not model capability, was
  the recurring root cause across multiple debugging sessions.
- **Idempotent ingestion by construction.** Chunk IDs are deterministic hashes of `doc_id` +
  slugified section title, and storage uses `upsert`, not `add` — re-running ingestion on
  unchanged docs is a safe no-op rather than a source of duplicate vectors. Upserting alone never
  *removes* anything, though — a real duplicate-document incident (the same policy ingested once
  as Markdown and once as a differently-`doc_id`'d PDF) surfaced that gap directly, which is why
  `PolicyVectorStore.delete_document(doc_id)` exists: it clears every parent/child chunk under a
  doc_id before a rename/replace/removal leaves orphaned vectors behind.
- **Rebuilt the golden dataset against the real corpus instead of trusting stale fixtures.**
  When the sample policy docs were swapped for a real 38-page government HR/admin PDF, the
  existing 24-example golden dataset kept referencing `doc_id`s (`hr-leave-policy`,
  `it-security-policy`) that no longer existed in the store — every IT-security question was
  silently unanswerable by construction, not because of a retrieval bug. Every one of the 49
  replacement examples was written by reading the actual ingested parent chunks back out of
  Chroma first, so `expected_doc_id`/`expected_section` are ground truth, not guesses.
- **A regex-based intent router had a silent blind spot for self-introductions.** Typing "my
  name is Harshil" skipped the greeting path entirely — `is_greeting()` only matched a fixed set
  of greeting words, so the message fell through to full RAG retrieval, got paraphrased by the
  query-rewrite LLM into something search-shaped, and returned a confidently-cited but nonsensical
  answer about the onboarding/induction policy. Fixed by extending intent classification to
  recognize self-introduction phrasing (`"my name is X"`, `"I'm X"`, `"call me X"`) and routing it
  through the same no-retrieval greeting path, now with a personalized reply.
- **The eval suite caught a citation-format bug that unit tests couldn't.** Reading stage-2
  answers turned up `gpt-oss-20b` occasionally writing citations with fullwidth lenticular
  brackets — `【source: hr-policy-dec-2025, Part 46】` instead of the requested
  `[source: ...]`. The citation parser only recognized square brackets, so those answers
  silently showed no sources and skipped citation validation. The parser now accepts both, with
  a regression test built from the real answer.
- **Read the judge's reasoning before trusting a failing score — again.** Two stage-2 failures
  looked like generation bugs until the DeepEval judge's reasons were printed: it had misparsed
  "The PAO, NHSRC" as two people and called a condition from the golden answer itself
  "irrelevant". They're recorded as known judge false negatives rather than "fixed" by tuning the
  eval until it passes.


