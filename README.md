# PolicyGuard

[![CI](https://github.com/parthivbhesaniya/policy_guard/actions/workflows/ci.yml/badge.svg)](https://github.com/parthivbhesaniya/policy_guard/actions/workflows/ci.yml)

**An agentic, self-correcting, evaluated RAG system for enterprise HR policy Q&A.**

🔗 **Live demo:** [policyguard-parthiv.streamlit.app](https://policyguard-parthiv.streamlit.app)

PolicyGuard answers employee questions strictly from a company's internal policy documents —
with forced, verified citations, an LLM hallucination check that retries before answering,
human-in-the-loop escalation with real checkpointing for anything it can't verify, and an
evaluation suite to measure all of it.

## Contents

- [Why this project exists](#why-this-project-exists)
- [Architecture](#architecture)
- [Key features](#key-features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Usage](#usage)
- [PDF support](#pdf-support)
- [Managing policy documents](#managing-policy-documents)
- [API & UI](#api--ui)
- [Deploying on Streamlit Community Cloud](#deploying-on-streamlit-community-cloud)
- [Docker](#docker)
- [Testing](#testing)
- [Observability (LangSmith)](#observability-langsmith)
- [Evaluation & results](#evaluation--results)
- [Engineering highlights](#engineering-highlights)

## Why this project exists

Internal HR questions ("how much sick leave do I get?") are repetitive, but wrong answers on
policy carry real compliance risk. A chatbot that makes something up when unsure is worse than
useless here — it has to know when it doesn't know, prove every claim against a real source, and
hand off to a human rather than guess. Every stage of PolicyGuard exists to make wrong answers
hard to produce and easy to catch.

## Architecture

![PolicyGuard StateGraph Architecture](./graph_structure.png)

Policy docs (`data/policies/*.md`, `*.pdf`) are chunked into parent/child sections and stored in
Chroma, with a BM25 index built over the same child chunks. Each question then runs through a
LangGraph `StateGraph`:

- `rewrite_query` resolves follow-ups into a standalone query. Its outgoing edge also checks
  intent, so greetings, off-topic and overly vague questions are answered without retrieval.
- `retrieve` runs dense + BM25 search, fuses the results, and reranks them with Cohere.
- `grade_documents` drops irrelevant excerpts; if nothing survives → `cannot_answer`.
- `generate` writes an answer with forced citations; `verify_answer` checks it is grounded and
  loops back to `generate` while retries remain.
- When retries run out, `escalate_to_human` pauses the graph (`interrupt()` + SQLite checkpoint)
  until a human approves, edits, or rejects the answer.

## Key features

- **Structure-aware chunking.** Markdown is split by `##` (parent) and `###` (child) headings,
  keeping tables intact. PDFs are chunked by their recovered heading structure — one parent per
  policy topic (e.g. `Leave Rules (Assistant Level Staff) › Casual Leave`) linked to ~600-char
  child chunks used for retrieval.
- **Intent routing before retrieval.** Greetings (including self-introductions like *"my name is
  Harshil"* → *"Hi Harshil!"*), off-topic questions, and vague ones (*"tell me the policy"*) get
  an instant reply instead of a search.
- **Hybrid retrieval.** Dense (Chroma) and BM25 keyword search are fused with Reciprocal Rank
  Fusion, then reranked by Cohere, so an exact keyword match doesn't lose to a
  similar-but-wrong chunk.
- **Corrective RAG grading.** An LLM filters retrieved sections down to the relevant ones; if
  none survive, the answer is "I don't have enough information" rather than a guess.
- **Validated citations.** Every claim must carry a `[source: doc_id, section]` tag, and each
  citation is checked against the retrieved context — a citation to unretrieved content counts
  as a grounding failure.
- **Safe conversational follow-ups.** Prior turns are used only to rewrite *"does this apply to
  interns?"* into a standalone query. `generate` never sees the chat history, so every claim
  still has to come from retrieved policy text.
- **A hallucination check that acts.** A second LLM pass verifies the answer is supported; if
  not, the graph regenerates (bounded retries) instead of shipping a low-confidence answer.
- **Real human-in-the-loop.** Exhausted retries pause the graph via LangGraph `interrupt()` and
  persist state to SQLite under a `thread_id`; a *separate process* resumes it later.
- **Evaluation.** A 49-example golden dataset (30 policy sections + 12 unanswerable questions)
  is scored by a four-metric `run_eval` harness and by a three-stage **DeepEval** suite judged
  by a larger model than the generator — see [Evaluation & results](#evaluation--results).
- **Fully dependency-injected.** LLM, retriever, reranker, and checkpointer are all swappable,
  so all 142 tests run against fakes in ~20 seconds with no API calls.

## Tech stack

| Layer | Technology | Role |
| --- | --- | --- |
| LLM | **Groq** via `langchain-groq` (`openai/gpt-oss-20b` default, `GROQ_MODEL`) | Every LLM call — rewrite, grade, generate, verify, judge — through one swappable `BaseChatModel` |
| Orchestration | **LangGraph** `StateGraph` | Explicit graph with conditional routing and a real verify → retry cycle |
| Vector search | **ChromaDB** | Parent/child collections. Embeds with HuggingFace `BAAI/bge-small-en-v1.5` when `HUGGINGFACE_API_KEY` is set, else Chroma's local `all-MiniLM-L6-v2` |
| PDF ingestion | **pypdf** + **RapidOCR** | Text extraction; near-empty (scanned) pages fall back to `pypdfium2` + `rapidocr-onnxruntime` OCR |
| Keyword search | **rank-bm25** | Sparse retrieval over the same child chunks, catching exact terms embeddings miss |
| Fusion | Reciprocal Rank Fusion (hand-rolled) | Merges dense and BM25 rankings |
| Reranking | **Cohere** Rerank | Reranks fused candidates (a local cross-encoder wasn't installable — see [highlights](#engineering-highlights)) |
| Human-in-the-loop | LangGraph `interrupt()` / `Command(resume=...)` + **langgraph-checkpoint-sqlite** | Pause mid-graph, resume from a different process |
| Tracing | **LangSmith** | Per-node traces from UI, API, CLI and evals, with outcome and 👍/👎 feedback |
| RAG evaluation | **DeepEval** (judge: Groq `openai/gpt-oss-120b`) | Retrieval, generation and end-to-end metrics as pytest tests |
| API / UI | **FastAPI** / **Streamlit** | HTTP wrapper around the graph / self-contained chat UI |
| Testing | **pytest** + **ruff** | 142 tests built on fakes (`FakeLLM`, `FakeCohereClient`) and temporary Chroma stores |
| Containers | **Docker** / Compose | `api` + `ui` services and a one-off `ingest` profile from one image |

## Project structure

```
policyguard/
├── data/
│   ├── policies/                    # Policy docs to ingest (Markdown + YAML front matter, or PDF)
│   └── eval/
│       ├── golden_dataset.json      # 49-example golden Q&A set
│       └── generation_golden.json   # ideal context per answerable question (verbatim Chroma chunks)
├── src/policyguard/
│   ├── ingestion/                   # loader.py (Markdown), pdf_loader.py (PDF text + OCR + sidecar YAML),
│   │                                # pdf_structure.py (heading recovery), chunker.py, vectorstore.py,
│   │                                # ingest.py (CLI: ingest / raw retrieval query)
│   ├── generation/                  # Baseline linear chain: chain.py, prompts.py, citations.py, ask.py (CLI)
│   ├── retrieval/                   # bm25.py, hybrid.py (RRF), reranker.py (Cohere)
│   ├── orchestration/               # state.py, nodes.py (nodes, intent checks, routing), graph.py,
│   │                                # ask.py (CLI), resolve.py (CLI: resume a paused thread)
│   ├── evaluation/                  # dataset.py, evaluators.py, run_eval.py (CLI), deepeval_judge.py
│   ├── observability.py             # LangSmith trace labels, outcome + 👍/👎 feedback
│   ├── api/                         # app.py (/ask, /ask/stream, /resolve, /health), schemas.py
│   └── ui/                          # app.py (Streamlit chat, runs the graph in-process), errors.py
├── evals/                           # DeepEval suite (live, billed LLM calls — not run by `pytest`)
│   ├── conftest.py                  #   shared fixtures: judge, store, retriever, reranker, generator
│   ├── test_retrieval.py            #   stage 1: Contextual Recall + Precision
│   ├── test_generation.py           #   stage 2: Faithfulness + Answer Relevancy (ideal context)
│   └── test_pipeline.py             #   stage 3: RAG triad over the full LangGraph app
└── tests/                           # 142 unit tests, one file per module
```

## Getting started

```bash
git clone <repo-url> && cd policyguard
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui]"   # drop `,ui` if you don't need Streamlit; add `,eval` for DeepEval
cp .env.example .env         # then add your API keys
```

| Variable | Needed for | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | Everything past ingestion | Free tier at [console.groq.com](https://console.groq.com/keys) |
| `GROQ_MODEL` | — | Default `openai/gpt-oss-20b` |
| `COHERE_API_KEY` | Reranking (or `--no-rerank`) | Free tier at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) |
| `COHERE_RERANK_MODEL` | — | Default `rerank-v3.5` |
| `HUGGINGFACE_API_KEY` | — | Optional HuggingFace embeddings (`HF_TOKEN` also works). Ingest and query with the same setting — vectors from different models don't mix |
| `LANGSMITH_TRACING` | — | **`.env.example` ships with `true`**; set `false` to disable tracing |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | Tracing, `run_eval --langsmith` | See [Observability](#observability-langsmith) |
| `POLICYGUARD_ENV` | — | Labels traces by deployment (e.g. `demo`); default `local` |
| `DEEPEVAL_JUDGE_MODEL` | — | Default `openai/gpt-oss-120b` |
| `EVAL_INCLUDE_REASON` | — | `1` makes DeepEval record the judge's reasoning (off to save tokens) |

Put your policy docs (Markdown or PDF) in `data/policies/` and ingest them:

```bash
python -m policyguard.ingestion.ingest --input data/policies --persist-dir ./chroma_db
```

## Usage

```bash
# Ask through the full graph (hybrid retrieval, grading, hallucination check)
python -m policyguard.orchestration.ask "how many carryover leave days am I allowed"
python -m policyguard.orchestration.ask "..." --no-rerank   # skip Cohere, no key needed
python -m policyguard.orchestration.ask                     # interactive mode, remembers follow-ups

# Resume a paused (escalated) thread from a separate process
python -m policyguard.orchestration.resolve --thread-id <id> --action approve
python -m policyguard.orchestration.resolve --thread-id <id> --action edit --answer "..."
python -m policyguard.orchestration.resolve --thread-id <id> --action reject

# Debugging aids
python -m policyguard.ingestion.ingest --query "..."   # raw retrieval only, no LLM
python -m policyguard.generation.ask "..."             # baseline chain, no grading/self-correction

# UI and API (independent of each other)
streamlit run src/policyguard/ui/app.py
uvicorn policyguard.api.app:app --reload

# Evaluation
python -m policyguard.evaluation.run_eval               # local scorecard (add --langsmith to track)
deepeval test run evals/test_retrieval.py               # stage 1 (also test_generation.py, test_pipeline.py)
deepeval test run evals/test_pipeline.py -k "office-hours-01 or tor-approval-01"   # a subset

# Reset the vector store
python -c "import shutil; shutil.rmtree('./chroma_db', ignore_errors=True)"
```

If an answer can't be verified after retries, `ask` prints a `thread id` and pauses instead of
answering. The DeepEval suite needs `pip install -e ".[eval]"` and an ingested `chroma_db`.

## PDF support

`--input` can mix `.md` and `.pdf` files; the file type is detected by extension.

- **Metadata** comes from an optional sidecar YAML with the same name (`finance_policy.pdf` +
  `finance_policy.yaml`) holding the same four fields as Markdown front matter:
  ```yaml
  doc_id: finance-expense-policy
  department: Finance
  effective_date: 2026-01-01
  version: 1.0
  ```
  If a sidecar exists, all four fields are required. Without one, metadata is guessed from the
  filename (slugified `doc_id`, keyword-guessed `department`, today's date, version `1.0`) and a
  warning is printed.
- **Chunking** ([pdf_structure.py](src/policyguard/ingestion/pdf_structure.py)) recovers parts
  (`PART V`), numbered sections (`7. Leave Rules`), lettered topics (`(b) Work From Home (WFH).`)
  and titled clauses (`(ii) Casual Leave.`), after stripping page headers, page numbers, the
  table of contents and look-alike Cyrillic/Greek letters. Each topic becomes a parent chunk (up
  to ~2,000 chars, longer ones split into `(cont. N)` parts) and is cited by its last two heading
  levels: `[source: hr-policy-dec-2025, Leave Rules (Assistant Level Staff) › Casual Leave]`. A PDF
  with no recoverable headings falls back to fixed-size windows cited as `Part N`.
- **Text extraction** uses `pypdf`; pages with almost no text (scans) are OCR'd instead.

## Managing policy documents

Ingestion only `upsert`s: it adds and updates, but never removes anything.

| You changed... | What to do |
| --- | --- |
| Content inside an existing section | Just re-run ingestion. Chunk IDs hash `doc_id` + section title, so chunks are overwritten in place |
| A section heading, or removed a section | Re-run ingestion, then delete the orphaned old chunk (its ID no longer matches) |
| A `doc_id` (including a renamed PDF with no sidecar) | Delete the **old** `doc_id` |
| Removed a policy file | Delete its `doc_id` |

There's no cleanup CLI yet:

```bash
python -c "
from collections import Counter
from pathlib import Path
from policyguard.ingestion.vectorstore import PolicyVectorStore

store = PolicyVectorStore(Path('./chroma_db'))
print(Counter(m['doc_id'] for m in store.get_all_children()[2]))   # what's in the store
store.delete_document('the-old-doc-id')                            # remove one document
"
```

## API & UI

The FastAPI service wraps the same compiled graph as the CLI. Chroma, BM25, the LLM and the
checkpointer are set up once at startup and shared across requests.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/ask` | POST | `{"question": "...", "history": [...]}` → `{"status": "answered" \| "cannot_answer" \| "needs_review", "answer": ..., "citations": [...], ...}` |
| `/ask/stream` | POST | Same request; server-sent events `status`, `token`, `error`, then a `final` event with the `/ask` body |
| `/resolve` | POST | `{"thread_id": ..., "action": "approve" \| "edit" \| "reject", "answer": ...}` — resumes a `needs_review` thread |
| `/health` | GET | Liveness check |

`history` is optional (`[{"question": ..., "answer": ...}, ...]`, oldest first). The API is
otherwise stateless, so the caller resends history with each follow-up. A `needs_review` thread
stays checkpointed until any client calls `/resolve`.

The **Streamlit UI doesn't use the API**: it builds the graph in its own process, keeps its own
chat history, and shows approve/edit/reject buttons when a thread pauses. It ingests
`data/policies/` on first load if the store is empty, and keeps review checkpoints in memory
(`InMemorySaver`), so a paused thread lasts only as long as the UI process.

## Deploying on Streamlit Community Cloud

Because the UI is self-contained, it deploys as a single app:

1. Create an app at [share.streamlit.io](https://share.streamlit.io) from this repo with main
   file `src/policyguard/ui/app.py`.
2. Add **Secrets**:
   ```toml
   GROQ_API_KEY = "..."
   GROQ_MODEL = "..."       # must be a model your Groq key can use -- see below
   COHERE_API_KEY = "..."   # optional; reranking is skipped if omitted
   # optional -- LangSmith tracing + 👍/👎 feedback
   LANGSMITH_TRACING = "true"
   LANGSMITH_API_KEY = "..."
   LANGSMITH_PROJECT = "..."
   POLICYGUARD_ENV = "demo"
   ```
3. Deploy. [requirements.txt](requirements.txt) (`-e .[ui]`) and [packages.txt](packages.txt)
   (apt libs for OCR) are picked up automatically.

The filesystem is ephemeral, so the app re-ingests `data/policies/` once per restart (cached
with `st.cache_resource`), and paused review threads don't survive a restart. Groq's model list
varies by key, so check yours before deploying — and check the model's output-tokens-per-minute
limit, since a low cap shows up as HTTP 429 on longer cited answers:

```bash
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY" | jq -r '.data[].id'
```

## Docker

```bash
touch checkpoints.sqlite   # otherwise Docker mounts it as a directory and the API can't open it
docker compose up -d api ui                       # API on :8000 (/docs), UI on :8501
docker compose --profile tools run --rm ingest    # (re-)ingest inside a container
docker compose down
```

- [Dockerfile](./Dockerfile) is a single `python:3.11-slim` stage. Every dependency ships a
  prebuilt wheel, so no compiler is needed; only `libgl1`/`libglib2.0-0` are installed for OpenCV
  (used by the OCR fallback).
- [docker-compose.yml](./docker-compose.yml) builds `api`, `ui` and `ingest` from one image. All
  three bind-mount `chroma_db/` and `~/.cache/chroma` (so Chroma's ~79 MB embedding model isn't
  re-downloaded per container); `api` also mounts `checkpoints.sqlite` and has a `/health`
  healthcheck.
- `api` and `ui` are independent — `docker compose up -d ui` alone gives a working chat UI.

## Testing

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

Both run on every push via [GitHub Actions](.github/workflows/ci.yml). `pytest` collects only
`tests/`: 142 tests, ~20 seconds, zero live API calls (LangSmith tracing is forced off in
`tests/conftest.py`). The DeepEval suite in `evals/` runs separately.

| File | Tests | Covers |
| --- | --- | --- |
| `test_orchestration.py` | 37 | Every node and routing decision, history resolution, full-graph interrupt/resume via `FakeLLM` |
| `test_pdf_ingestion.py` | 21 | PDF topic chunking + window fallback, sidecar validation, guessed defaults |
| `test_observability.py` | 17 | Trace labels, outcome classification, feedback, no-op when tracing is off |
| `test_evaluation.py` | 16 | Dataset integrity, programmatic and LLM-judge metrics |
| `test_pdf_structure.py` | 12 | PDF heading recovery and cleanup, word-boundary splitting |
| `test_retrieval.py` | 9 | BM25, RRF fusion, Cohere reranker (fake client) |
| `test_ui_errors.py` | 9 | Groq/Cohere errors → friendly UI messages |
| `test_chunker.py` | 8 | Markdown hierarchical chunking, parent/child linking |
| `test_chain.py` | 5 | Context deduping, prompt construction |
| `test_citations.py` | 5 | Citation parsing (incl. fullwidth `【source: …】`) + validation |
| `test_vectorstore.py` | 2 | `delete_document` |
| `test_api_streaming.py` | 1 | `/ask/stream` event shape |

The orchestration tests run the **actual compiled LangGraph app**, including the
interrupt/checkpoint/resume cycle, with only the LLM faked — so the graph wiring itself is tested.

## Observability (LangSmith)

With `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` set, every question
becomes a LangSmith trace with a child run per graph node — prompt, output, tokens, cost and
latency — so a wrong answer can be traced to retrieval, grading or generation.
[observability.py](src/policyguard/observability.py) adds:

| Signal | Recorded as | Values |
| --- | --- | --- |
| Source | tags + metadata | `entrypoint`: `ui` / `api` / `cli` / `eval`; `environment`: `POLICYGUARD_ENV` |
| How it ended | feedback `outcome` | `answered`, `cannot_answer`, `escalated`, `human_reviewed`, `greeting`, `out_of_scope`, `clarification` |
| Self-correction | feedback `retries` | generation attempts beyond the first |
| Citation problems | feedback `invalid_citations` | citations to sections that weren't retrieved |
| User rating | feedback `user_score` | 👍 = 1 / 👎 = 0 from the Streamlit UI |

![LangSmith trace of a live demo question: the node waterfall, with the user's 👍 and the outcome, retries, and invalid_citations feedback](./Lang_Smith_Dashboard/LS_Dashboard.jpeg)

*A live demo question: per-node timing (2.26 s, ~3.9K tokens, $0.0005 total) and the user's 👍
next to the automatic `outcome`, `retries` and `invalid_citations` feedback.*

With tracing off, every helper is a no-op and the 👍/👎 buttons are hidden. Feedback is sent from
a background thread, LangSmith outages are logged rather than raised, and the UI tells users
their questions are logged whenever tracing is on.

## Evaluation & results

The DeepEval suite in [evals/](evals/) scores each component separately, then the whole
pipeline, so a bad score points at a specific stage. It uses the 37 answerable golden questions,
a pass mark of **0.7** (Contextual Relevancy is report-only), and `openai/gpt-oss-120b` as judge —
larger than the `openai/gpt-oss-20b` generator. Results from 2026-10-02:

| Stage | What runs | Metrics | Result (37 questions) |
| --- | --- | --- | --- |
| 1. Retrieval | `retrieve` alone (hybrid → rerank → top 4) | Contextual Recall, Precision | **36 / 37** (recall 1.0 on all) |
| 2. Generation | `generate` alone, on *ideal* context from [generation_golden.json](data/eval/generation_golden.json) | Faithfulness, Answer Relevancy | **35 / 37** (both judge errors) |
| 3. Pipeline | Full graph: rewrite → retrieve → grade → generate → verify | Faithfulness, Answer Relevancy (+ Contextual Relevancy) | **37 / 37** — Faithfulness ≥ 0.8, Answer Relevancy ≥ 0.75 everywhere |

What the failures mean:

- **Retrieval, `contract-termination-notice-01`** (Precision 0.5): the question mentions
  "probation", so the Probation chunk outranks the right one (rank 2). Stage 3 still answers it
  perfectly.
- **Generation, two judge false negatives:** both answers were checked by hand and are correct
  and fully grounded; the judge gave Answer Relevancy 0.67. Which questions trip it varies by run
  (documented in [test_generation.py](evals/test_generation.py)).
- **Contextual Relevancy is low (mean 0.32) by design of the metric:** it scores the share of
  *sentences* relevant to the question, so "When is a medical certificate required?" scores 0.25
  against the perfect four-sentence Sick Leave clause. It's tracked, not used as pass/fail.

**Before/after: fixed windows → topic chunks.** The first run used ~2,000-char fixed `Part N`
windows that mixed several topics. Rebuilding the PDF chunker around the document's headings
gave:

| | Fixed windows | Topic chunks |
| --- | --- | --- |
| Parent chunks (median size) | 79 (~1,730 chars) | 112 (~640 chars) |
| Citations | `Part 34` | `Leave Rules (Assistant Level Staff) › Sick Leave` |
| Context per question (5-question sample) | ~2,470 chars | ~790 chars (**−68%**) |
| Stage 1 / Stage 2 | 36 / 37, 35 / 37 | 36 / 37, 35 / 37 |
| Stage 3 Faithfulness / Answer Relevancy | 1.0 / ≥ 0.75 | ≥ 0.8 / ≥ 0.75 |
| Stage 3 Contextual Relevancy | 0.06–0.62 | 0.09–1.0, mean 0.32 |

Same answer quality with two-thirds less context (so fewer tokens and lower latency) and
human-readable citations.

Running notes: the suite is kept out of `pytest` (`testpaths = ["tests"]`) because every metric
is a billed judge call. The judge retries rate limits, and stage 3 stops cleanly when Groq's
daily quota runs out, so the rest can be resumed with `-k`. On the free tier (200K tokens/day)
the full suite takes a few days. Stage 2 fails loudly if re-ingestion changes the pinned ideal
context.

## Engineering highlights

Lessons from running the system against a live LLM:

- **Bad citations now fail verification.** The model sometimes cited a `###` heading seen inside
  an excerpt instead of the excerpt's own source. Fixed in the prompt, and citation validity now
  feeds the groundedness check, so the retry loop catches it.
- **Judge bugs, not system bugs.** Surfacing the judge's reasoning showed it was failing correct
  "I don't know" answers and penalizing properly declined off-topic questions. Both judge
  prompts were fixed.
- **Adapted to a platform constraint.** The planned local cross-encoder needs `torch`, which has
  no wheel for Intel Macs on Python 3.13. Reranking moved to Cohere behind the same interface,
  with a `--no-rerank` escape hatch.
- **Follow-ups needed the rewritten query everywhere.** With chat history wired into
  `rewrite_query`, *"does this apply to interns"* still failed because grading and generation saw
  the raw question. Both now receive `rewritten_query`.
- **Data hygiene was the recurring root cause.** The same follow-up also failed because the store
  held a second organization's HR manual under another `doc_id`, and an earlier incident ingested
  one policy twice (as Markdown and PDF). Upsert-only ingestion never removes anything, which is
  why `delete_document(doc_id)` exists.
- **Golden dataset rebuilt from the real corpus.** After switching to the real 38-page HR PDF, the
  old dataset referenced `doc_id`s that no longer existed. All 49 new examples were written from
  chunks read back out of Chroma.
- **Self-introductions skipped the greeting path.** *"my name is Harshil"* went to retrieval and
  returned a confidently cited, nonsensical answer. Intent checks now recognize introductions.
- **The eval suite caught a parser bug.** `gpt-oss-20b` sometimes wrote fullwidth
  `【source: …】` citations, which the parser silently ignored. It now accepts both, with a
  regression test.
- **Judge false negatives are recorded, not tuned away.** Two stage-2 failures came from the
  judge misreading answers; they're documented rather than "fixed" by bending the eval.
