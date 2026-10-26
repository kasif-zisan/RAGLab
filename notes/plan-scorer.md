# Plan: the scorer (next build step) — written 2026-10-14

## Why it comes before any index

Twelve sweep points are meaningless without the thing that ranks them, and a bug
in the scorer silently invalidates all twelve. Build and validate it first.

## Two separate measurements — do not conflate them

**1. Chunking metric (free, no embedding, no index, no money).**
For each of the 393 questions: does ANY single chunk contain the complete gold
span? This is the **retrieval ceiling**. If the evidence did not survive
chunking, no retriever, reranker or prompt can recover it. Run this on every
sweep point before spending anything on embeddings — it may eliminate candidates
for free.

**2. Retrieval metric (needs an index).**
recall@k: embed the question, retrieve top-k, check whether a retrieved chunk
contains the gold span. Bounded above by metric 1.

## The containment test

Reuse `row_survived` from Phase 0 (`infra/job/run.py`): every cell of a gold row
must appear on ONE line, in order. Already validated — it scored 4/4 on the
ORLY table and is strict enough not to pass on scattered cells.

For prose spans (non-table), fall back to normalised substring containment
(collapse whitespace, then `in`).

**Caveat that must be handled:** gold spans are the benchmark's compact markdown.
Arm A holds docling's padded markdown (measured median 2.12x larger, p90 5.40x);
arms B1/B2 hold flat text with no pipes at all. So a literal string match will
fail on every arm. The test must be **cell-level, order-preserving, and
format-insensitive** — compare the sequence of cell values, not the rendering.

## Per-arm expectations worth checking

- Arm A should score high on table questions (headers present, 0% headless).
- Arms B1/B2 lose the column header entirely, so a question needing "which year"
  may have its cells present but unanswerable. The metric measures presence, not
  answerability — state that limit rather than overclaim.

## Join key

`qa.parquet.file_name` -> chunk `metadata["source"]`. 393 questions, 196 gold
docs, every corpus doc is gold.

## Output shape

One row per (arm, chunk_size, question): contained yes/no, which chunk, how many
chunks the span was spread over. Aggregate to a containment rate per sweep point.
