# Query-time cost and latency — the number that never stops

Written 2026-10-30, measured on this project.

Index-time cost is **one-time**. Query-time cost is **forever**. Most of the
embedding notes are about the one-time half; this one is about the half the
client lives with.

---

## 1. What one query actually costs us

Two components, both measured:

| step | measured | notes |
|---|---|---|
| embed the question | **~38 ms** on 8 vCPU CPU | 393 questions, Qwen3-0.6B, from today's run |
| search arm B1 (37,873 vecs) | **9.2 ms** | exact brute force, one matmul |
| search arm A (99,265 vecs) | **26.8 ms** | |
| search both (137,138 vecs) | **37.0 ms** | |

So **retrieval end-to-end is roughly 50 ms** for B1, on CPU, with no vector
database and no ANN index. On a GPU the embed step drops to single-digit
milliseconds.

That is worth internalising before anyone proposes infrastructure: at this
corpus size, exact search over flat files is not the slow part of anything.

---

## 2. Put it in context — retrieval is not the expensive step

A full RAG request is roughly:

    embed the question        ~38 ms      <- this note
    retrieve top-k            ~9-37 ms    <- this note
    rerank (if used)          ~50-300 ms
    LLM generates the answer  ~1,000-10,000 ms   <- dominates everything

**Generation is 20-200x the entire retrieval path.** Optimising a 9 ms search
down to 3 ms is invisible to the user and irrelevant to the bill.

Say this to clients early. It redirects the conversation from "which vector
database is fastest" — which almost never matters — to answer quality and token
cost, which always do.

---

## 3. The recurring cost, and where it actually lands

Per query, if you use a managed embedding API:

    a question is ~25 tokens
    at ~$0.02 per million input tokens  ->  $0.0000005 per query
    1,000,000 queries                   ->  about $0.50

**Query embedding is free in practice.** It is a rounding error next to
generation tokens. Do not let it into a cost model as a line item.

What actually recurs:

| cost | shape | at our size |
|---|---|---|
| query embedding | per query | negligible |
| vector storage | per month | 281 MB, ~$0.01/mo on S3 |
| LLM generation | per query | the entire bill |
| re-indexing | per rebuild | $0.80 per full rebuild |
| a vector database | **per hour, whether used or not** | $0 for us; $70-500+/mo managed |

Note the fourth row. A managed vector database is the only item here that bills
while nobody is asking questions. At our scale that is the whole argument
against adopting one early.

---

## 4. If you use a managed embedding API, latency changes shape

Self-hosted, the query embedding is local compute: ~38 ms CPU, less on GPU.

Through an API it becomes a **network round trip plus queueing**, typically
100-500 ms, and from Bangladesh to a US region considerably worse. That can be
10x the local cost and it is latency you cannot optimise away.

Three consequences worth planning for:

- **Pick a region near your users**, not near you. This is a bigger latency
  decision than anything in your code.
- **The embedding API is now a dependency on your critical path.** If it is
  throttled or down, nobody can search. Budget a timeout and decide what
  happens on failure — degrade to keyword search, or return an error.
- **Cache query vectors.** Real user traffic repeats: the same questions get
  asked constantly. A small cache keyed on the normalised question text removes
  that round trip entirely for repeat queries.

---

## 5. What to measure before quoting a latency SLA

Measure these on the client's data and the client's region, not yours:

1. **p50 and p95 query embedding latency** — the mean hides the tail, and the
   tail is what people complain about.
2. **Search latency at the client's real index size**, not at your demo size.
   It scales roughly linearly in brute force.
3. **The full path**, end to end, including generation. That is the number the
   user experiences and the only one worth an SLA.
4. **Under concurrency.** All of the above change once ten people query at once,
   and a single-threaded numpy matmul does not scale the way a service does.

Quote the end-to-end p95, not the retrieval number. Retrieval is a part you
control; generation latency is largely the model provider's, and you should say
so in writing before agreeing to anything.
