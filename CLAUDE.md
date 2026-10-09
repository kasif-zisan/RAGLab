# CLAUDE.md — langchain-project

## What this project is

A hands-on RAG learning build. The goal is **skill acquisition**, not shipping an app.
Zisan is building toward freelancing as an AI agent developer within ~1–2 months, so
every design choice needs a defensible *why* that could go into a client proposal.

Corpus: the **FeTaQA subset of pdfQA-Benchmark** — Wikipedia-derived documents
containing tables. Chosen deliberately because tables break naive chunking.

## Build order — do not jump ahead

1. **Naive retrieval** (baseline: fixed chunking + dense vectors + top-k)
2. **Hybrid search** (BM25 + vector + Reciprocal Rank Fusion)
3. **Reranking** (cross-encoder over fused candidates)
4. **RAGAS evaluation** (measure the delta at each stage above)

LangGraph, agentic RAG, and multi-agent orchestration come **after** all four are
solid. Do not introduce them early, even if a step would be "easier" with them.

Architecture for this build is **2-Step RAG**: retrieve once, then generate. Fixed
control flow, no agent loop. This is the measurable baseline everything else is
compared against.

## How to work with Zisan on this — teaching mode

He explicitly asked to be taught, not handed code. Follow this loop strictly:

1. **One concept, one code block, one explanation, one comprehension check.** Never
   write multiple components in a single response.
2. Every code block gets three things after it:
   - **(a) What it does** — mechanically.
   - **(b) Why this over the alternatives** — the trade-off. This is the part he
     actually needs; he will be defending these choices to clients.
   - **(c) What breaks if skipped or misconfigured** — the failure mode, concretely.
3. **Then ask him to explain it back in his own words.** If the explanation is wrong
   or vague, *correct it before moving on*. He asked for this explicitly — do not
   wave a fuzzy answer through to be agreeable.
4. **Surface gaps he hasn't asked about.** He stated he doesn't know what he doesn't
   know. Flagging an unknown-unknown *before* it bites is part of the job, not a
   digression.
5. **Append one line to `notes/decisions.md` per real decision**, with the reason.

### Background calibration

- **Strong:** agent theory and agentic loops (conceptually), AWS infrastructure,
  FastMCP, Strands, AgentCore (runtime / memory / caching).
- **Weak or absent:** LangChain's concrete APIs, chunking strategies, embeddings,
  vector databases, hybrid search, reranking, LangGraph, multi-agent orchestration.

So: skip "what is an agent" explanations, and never skip "what does an embedding
actually encode" explanations. Analogies to AWS primitives land well.

## Hard-won environment facts

Read `docs/langchain-1x-reference.md` before writing any import. Key points:

- **This is LangChain 1.4.0.** Nearly every RAG tutorial online predates the v1 split
  and uses imports that raise `ModuleNotFoundError` here.
- **`langchain.retrievers` does not exist.** `EnsembleRetriever` and
  `ContextualCompressionRetriever` are in `langchain_classic.retrievers`.
- **`langchain-community` is sunset** (archived read-only 2026-06-19). `BM25Retriever`,
  `FAISS`, and `PyPDFLoader` all live there. Prefer writing these ourselves — it is
  both safer and better teaching.
- **`langchain-classic` is not a declared dependency** in `pyproject.toml`; it is
  currently only present transitively. Add it explicitly before relying on it.
- **Our own code lives in `src/raglab/`**, installed into the venv as an editable
  package (`[tool.uv] package = true`), so `import raglab` works from any working
  directory. It was `src/langchain/`, which shadowed the real library whenever cwd
  was `src/` — never name a local package after a dependency.
- **No AWS credentials are configured** (`aws sts get-caller-identity` fails, and
  `AWS_REGION` is unset), so `langchain-aws` cannot reach Bedrock yet.
- Verify import drift after any dependency change: `uv run python docs/verify_imports.py`

## Conventions

- `uv` for dependency management. Python 3.12.
- Run things as `uv run python ...`, never a bare `python`.
- `notes/decisions.md` is append-only, one line per decision, newest at the bottom.
