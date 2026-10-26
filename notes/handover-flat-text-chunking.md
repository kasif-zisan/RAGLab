# Handover: flat-text chunking (arm B) — 2026-10-09

Read `CLAUDE.md` first (teaching rules), then this. `notes/decisions.md` has the
full decision log, newest at the bottom; the last ~60 entries cover today.

## What is happening right now

EC2 instance `i-0c43831a99ccecd64` is parsing all 196 PDFs with Docling.
Finishes ~2h47m after start, writes one JSON per document to
`s3://raglab-zisan-apne1/parsed/<stem>.json`. **Do not touch that instance or
that prefix.** Check it with `infra/watch.sh i-0c43831a99ccecd64`.

All AWS calls use `--profile zisan --region ap-northeast-1` and nothing else.
**Never install a dependency without asking Zisan first.**

## The experiment this work belongs to

Three arms, same eval, same chunk size, compared on retrieval recall@k:

| arm | representation | status |
|---|---|---|
| A | Docling markdown (layout-aware) | parsing now |
| B | **flat PyMuPDF text** | **this session's job** |
| C | both in one index | after A and B |

Arm B is the baseline that says what the parsing work was worth. If B ≈ A, the
Docling pipeline bought nothing measurable. That is the point of running it.

**Chunk size must be identical across all three arms.** Tuning it per arm changes
two variables and kills the comparison. Per-arm tuning is a follow-up, reported
separately.

## Arm B's starting point

`data/finqa/text_pymupdf/<stem>__pymupdf.txt` — 196 files, one flat string each,
78,008,165 chars total (~19.5M tokens). Shipped with the benchmark; we did not
produce it. `src/raglab/corpus.py` already loads it into LangChain `Document`s
with metadata (`source`, `ticker`, `year`, `is_gold`, `chars`).

Known properties, measured: **zero `\n\n` in the whole corpus** (6,956 `\n` in
ORLY_2015, 48.8 chars between line breaks on average). So the default
`RecursiveCharacterTextSplitter` separator list degrades straight to `\n`.

## First task, and it is a real gap

**Arm B has no page boundaries; arm A does.** Docling now emits one markdown
string per page, so arm A can cite "page 50". The shipped flat text is one blob
per document, so arm B can only cite the document.

That asymmetry does not break a chunk-level recall comparison, but it does mean
arm B cannot produce page citations — and citations are the feature that makes an
imperfect RAG system trustworthy.

Cheapest fix, measured: PyMuPDF flat extraction runs at **8.3 ms/page**, so
re-extracting all 23,000 pages *with page boundaries kept* takes **~3.2 minutes
on one core**. The PDFs are in `data/finqa/pdf/`. This is text extraction, not
model inference — it will not heat the laptop the way Docling would — but
**confirm with Zisan before running anything locally**, since the standing rule
is that compute goes to AWS.

## Things already decided — do not relitigate

- Metadata is attached at load time, not after splitting, so it propagates to
  every chunk. `metadata["view"]` = `"docling"` or `"flat"` distinguishes the arms (named after
  provenance, not content - the docling markdown contains prose too).
- The corpus list comes from `data/finqa/manifest.json`, never from globbing.
- `chunk_size` is a maximum, not a target, and `length_function=len` counts
  **characters, not tokens**.
- Embedding dimensions stay pinned at 1024 while chunk size is swept.
- Imports come from `langchain_core` / `langchain_text_splitters`. This is
  LangChain 1.4.0 — `langchain.retrievers` does not exist. Read
  `docs/langchain-1x-reference.md` before writing any import.

## Open question for this session

The right chunk size has never been measured. `data/finqa/qa.parquet` has a
`source_text` column holding the exact gold evidence span for each of 393
questions. **Measure that distribution and set the chunk-size floor from it**,
rather than borrowing 1000/200 from a tutorial. That was the next code block
before parsing took over, and it is still unstarted.
