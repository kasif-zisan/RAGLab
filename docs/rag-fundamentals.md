# RAG Evaluation Fundamentals

Written 2026-09-06, from zero. Terminology and measurement concepts for the
2-Step RAG build. Read before `notes/decisions.md` makes sense.

## The frame: an open-book exam

RAG is a student taking an open-book exam. There is a library. A question arrives.
The student **finds the right page**, then **writes an answer from that page**.

Two separate skills. Failing by grabbing the wrong book and failing by misreading
the right book produce the *same wrong answer* but need *opposite fixes*. That
distinction drives everything below.

## Vocabulary

| Term | Plain meaning | In this project |
|---|---|---|
| Corpus | The library — everything searchable | 878 FeTaQA documents |
| Query | The question asked | `question` column |
| QA pair | A question bundled with its known-correct answer | 69 of them |
| Eval set | All QA pairs together; the exam paper | those 69 |
| Ground truth / reference | The known-correct thing you grade against | `answer`, `source_text` |
| Gold document | The document that genuinely contains the answer | 67 of the 878 |
| Distractor | A document that does not answer this question | the other 811 |
| Chunk | A slice of a document (too big to feed whole) | we create these |
| Context | The chunks retrieved and passed to the LLM | pipeline output |
| k | How many chunks to fetch per question | our choice |

"Gold" is jargon for "the verified correct one" (from *gold standard*). Gold status
is **per question** — the Florida governors page is gold for a governors question
and a distractor for a Dolphins question.

### A real row

- **question**: Who appointed John Branch as territorial governor in 1844, and what year did he serve until?
- **answer**: In 1844, John Tyler appointed John Branch as Florida's territorial governor until 1845.
- **file_name**: `List of Governors of Florida`  <- the gold document
- **source_text**: the governors table  <- the reference context

## The two stages

```
question
   |
   +--> STAGE 1: RETRIEVAL    search 878 docs -> return k chunks  ("context")
   |                                                     |
   +--> STAGE 2: GENERATION   question + context -> LLM -> answer
```

Stage 1 is a search engine — no LLM writing anything. Stage 2 is an LLM reading
what stage 1 handed it.

## Why measurement splits along that seam

Suppose the pipeline answers "Andrew Jackson" — wrong. Two possibilities:

- **A** — retrieval returned the wrong document; the LLM never saw the table and guessed.
- **B** — retrieval returned the right table; the LLM read the wrong row.

Identical symptom, opposite fixes. A needs a better retriever (milestones 1-3);
B needs a better prompt or model. Grading only the final answer cannot tell them
apart — it is a system with no per-layer logging.

## The four metrics

**Retrieval** — retrieved chunks vs. `source_text`:

- **Context Recall** — of the information needed, how much was actually fetched?
  This is the **ceiling on everything downstream**: un-retrieved context cannot be
  used by any model.
- **Context Precision** — of what was fetched, how much was relevant? Retrieve 10
  chunks where 1 matters and precision is 0.1; the other 9 cost tokens and distract.

These trade off against each other. Raising `k` raises recall and lowers precision.
Choosing `k` is choosing a position on that curve — a decision, not a default.

**Generation** — the answer itself:

- **Faithfulness** — does the answer assert only what the context supports?
  Low faithfulness = hallucination.
- **Answer correctness** — does the answer semantically match the reference answer?

## The diagnostic table

| Context Recall | Faithfulness | Diagnosis |
|---|---|---|
| low | high | **Retriever.** LLM faithfully used bad context. Fix search. |
| high | low | **Generator.** Right context, model ignored or invented. Fix prompt/model. |
| high | high | Working. |
| low | low | Both — fix retrieval first, it is upstream. |

Example: recall 0.62 / faithfulness 0.94 is row one. The generator is fine; it is
being fed wrong pages 38% of the time. Fix the retriever — which is what
milestones 1-3 do.

This table is the reason `source_text` is kept. Without a reference context there
is no recall column, and the remaining number diagnoses nothing.

## Why grading is not string comparison

Reference: "In 1844, John Tyler appointed John Branch as Florida's territorial
governor until 1845."  Pipeline: "John Tyler, until 1845."  Correct — and exact
match scores it zero.

| Method | How | Suited to |
|---|---|---|
| Exact match | strings identical | short factual spans (`$18,398`, Tat-QA) |
| Token overlap (ROUGE/BLEU) | shared-word fraction | cheap, crude, punishes rewording |
| Semantic / LLM-as-judge | an LLM rules on equivalence | free-form prose (FeTaQA) |

FeTaQA's ~120-character prose answers require the third. **RAGAS** is a library
that computes the four metrics above using an LLM as judge — which costs money per
evaluation run, so eval-set size is a budget decision as well as a statistical one.

## Related

- `docs/langchain-1x-reference.md` — verified API surface
- `notes/decisions.md` — every choice made and why

## What `source_text` actually contains

A numpy array of strings, almost always length 1. The element is the reference
context in markdown pipe-table format. Across the 69 FeTaQA rows: 183 to 6926
characters, median 1119.

A complete example (row: `Xeon`, question "What are the Xeon models and their
processing speed (Ghz)?"):

```
| Model | Speed (GHz) | L2 Cache (MB) | FSB (MHz) | TDP (W) |
| --- | --- | --- | --- | --- |
| X3210 | 2.13 | 2 × 4 | 1066 | 100/105 |
| X3220 | 2.40 | 2 × 4 | 1066 | 100/105 |
| X3230 | 2.66 | 2 × 4 | 1066 | 100 |
```

The same table in `text_pymupdf/Xeon__pymupdf.txt` (lines 1167-1185) is flattened
column-header-first, one cell per line:

```
Speed (GHz)
L2 Cache (MB)
FSB (MHz)
TDP (W)
X3210
2.13
2 × 4
1066
100/105
X3220
...
```

The binding between `X3210` and `2.13` survives only as "five lines apart." A chunk
boundary between them separates the model from its clock speed permanently.

### How the reference is used in scoring

Context Recall is not string comparison. An LLM judge decomposes `source_text` into
atomic claims ("X3210 runs at 2.13 GHz"), then checks each against the retrieved
chunks. Recall is the fraction supported. Format mismatches do not hurt; missing
information does.

### `source_text` is a label, not ground physical truth

In the Xeon row the answer is *also* present in prose ~30 lines above the labeled
table ("The models are the X3210, X3220 and X3230, running at 2.13 GHz, 2.4 GHz and
2.66 GHz, respectively"), and the reference `answer` paraphrases that sentence
rather than the table. A retriever that finds only the prose answers correctly but
scores low context_recall.

Consequence: when context_recall and answer correctness disagree, inspect the
retrieved chunks before concluding the retriever is broken. Optimizing hard against
a benchmark label can chase an artifact.

### Provenance: why the reference is cleaner than the document

Verified 2026-09-06: the pipe-table format appears nowhere in
`text_pymupdf/Xeon__pymupdf.txt` (3 pipe chars in the whole file, all Wikipedia
citation cruft). The exact row `| X3210 | 2.13 | 2 x 4 | 1066 | 100/105 |` is not
present; the individual cell values are.

```
Wikipedia HTML table
    |
    +--------------------> source_text          (structured, clean)
    |
    +--> rendered to PDF --> pymupdf --> text_pymupdf/   (flattened)
```

FeTaQA was originally a *table* dataset. pdfQA-Benchmark rendered those tables into
PDFs. So the PDF is downstream of the table, and `source_text` is the original
structured form - not an excerpt of the document.

Consequence: a retriever can never return text matching `source_text`'s format,
because it searches the flattened extraction. This is fine - the judge compares
meaning, not characters - but do not engineer toward format alignment.

### Chunk vs. source_text

|  | Who creates it | Size | Stability |
|---|---|---|---|
| Chunk | We do, by splitting documents | our parameter | changes every experiment |
| `source_text` | Benchmark annotators | whatever the table was | fixed |

`source_text` is the yardstick; chunks are what is being measured. If both moved
there would be no measurement.

### Which metric needs which reference

| Metric | `source_text` | `answer` | retrieved context |
|---|---|---|---|
| Context recall | yes | - | yes |
| Context precision | yes | - | yes |
| Faithfulness | - | - | yes |
| Answer correctness | - | yes | - |

Faithfulness requires no reference at all - it asks only whether the generated
answer stays within what the retrieved context supports. It can therefore run on
unlabeled production traffic, while recall and correctness need a labeled eval set.

---

## Generalizing beyond FeTaQA: the shape of RAG datasets

### The universal skeleton

Every RAG evaluation dataset is an arrangement of four components:

| Component | Purpose | FeTaQA's version |
|---|---|---|
| Corpus | what is searched | 878 documents |
| Queries | what is asked | `question` |
| Relevance labels | which content is correct to *retrieve* | `file_name` + `source_text` |
| Reference answers | what a correct *answer* looks like | `answer` |

The third is the one beginners miss - question/answer pairs alone cannot measure
retrieval.

### Families

| Family | Examples | Shape |
|---|---|---|
| Classic IR | BEIR, MS MARCO, TREC | corpus + queries + qrels, **no answers** |
| Extractive QA | Natural Questions, SQuAD, TriviaQA | short answer spans + source passage |
| Multi-hop QA | HotpotQA, 2WikiMultihop, MuSiQue | **2+ gold docs**, supporting-fact labels |
| Long-form QA | FeTaQA, ELI5, ASQA | prose answers, semantic grading required |
| Domain document QA | FinanceBench, CUAD, PubMedQA, DocVQA | real business PDFs |
| Synthetic | LLM-generated from a corpus | what client work actually requires |

Classic IR layout:

```
corpus.jsonl    {"_id": "doc7", "title": ..., "text": ...}
queries.jsonl   {"_id": "q3", "text": "who appointed john branch"}
qrels.tsv       q3    doc7    1
```

### Vocabulary

| Term | Meaning |
|---|---|
| qrels | query relevance judgments - the label file |
| positive / negative | relevant / irrelevant doc (= gold / distractor) |
| hard negative | a distractor that *looks* relevant; same topic, wrong answer |
| graded relevance | relevance scored 0-3 rather than binary |
| @k | computed over the top k results, e.g. `Recall@10` |

**Hard negatives** matter most. FeTaQA's 811 distractors are mostly *easy* negatives
(a Dolphins question vs. a Xeon page separates trivially). A hard negative would be a
different Intel processor table with identical column headers and different clock
speeds. Real corpora are full of them - forty near-identical quarterly reports, each
correct for exactly one question - which is why benchmark-good retrieval can collapse
on a client corpus.

### Trap: "recall" names two different metrics

| | `Recall@k` (IR) | `context_recall` (RAGAS) |
|---|---|---|
| Unit | whole documents | atomic claims |
| Labels | binary qrels | LLM-judged against reference text |
| Asks | was the gold doc in the top k? | was the needed information covered? |
| Cost | free | one LLM call per question |

Always state which one is meant.

### Where FeTaQA is not representative

1. Exactly one gold document per question - multi-hop datasets need several, and real
   client questions often span documents.
2. `source_text` is a clean structured table, not an excerpt of the document - a
   consequence of the render-to-PDF provenance. Most datasets give a real passage.
3. Synthetic PDFs - no scans, OCR noise, rotated pages, or multi-column layouts.

### What client work actually requires

Clients have no questions, no answers, and no relevance labels - only a folder of
PDFs. The billable skill is **manufacturing an eval set from an unlabeled corpus**:
sample chunks, LLM-generate questions against each, keep the source chunk as the
relevance label, human spot-check a subset. RAGAS ships test-set generation for this.

FeTaQA is training wheels - a corpus whose labels already exist, so we learn what
good labels look like before having to produce them.
