# .npy files vs a vector database — when you need one


## 1. What we have, and what it is not

`.npy` is a **storage format**. It is not a search system.

Our search is brute force and **exact**: multiply the query vector by every
chunk vector, take the largest scores. Because the vectors are normalised,
cosine similarity and dot product rank identically, so it is literally one
matmul.

    scores = V @ q          # V is (137138, 1024), q is (1024,)
    top    = np.argpartition(-scores, k)[:k]

That gives you ranking and nothing else. No filtering, no updates, no server,
no concurrency, no persistence beyond the files.

---

## 2. What a vector database adds

1. **Approximate nearest neighbour (ANN) search** — HNSW, IVF, ScaNN. Trades
   exactness for speed, which is the only way past a few million vectors.
2. **Metadata filtering** — "only chunks from AAPL, fiscal 2015". Doing this
   properly *with* ANN is genuinely hard and is a real reason to buy rather than
   build.
3. **Incremental CRUD** — add, update, delete one document without rebuilding
   the whole index. Our shards are rebuild-only.
4. **Serving** — a query API, concurrency, auth, replication, backups.
5. **Hybrid search** — BM25 and vector scoring in one place, often with fusion
   built in.

Note that (1) is the one everyone talks about and (2) and (3) are usually the
ones that actually force the decision.

---

## 3. When you actually need one

Measured on this laptop, fp32, one query, top-10:

| vectors | size fp16 | brute-force query | verdict |
|---|---|---|---|
| **137,138 (ours)** | 0.28 GB | **38 ms** | brute force is fine |
| 1,000,000 | 2.05 GB | 295 ms | too slow interactive — want ANN |
| 10,000,000 | 20.5 GB | does not fit in RAM | need a real system |

Rough thresholds, worth sanity-checking on the actual hardware:

- **under ~1M vectors** — brute force, or an in-process ANN library. No server.
- **1M–10M** — you want an ANN index. Can still be in-process (FAISS, hnswlib).
- **over ~10M, or multi-tenant, or heavy filtering, or live updates** — a real
  vector database earns its keep.

Most RAG projects never leave the first row. 196 SEC filings produced 137k
chunks; a client with 2,000 documents is in the same place.

---

## 4. The options, cheapest commitment first

| kind | examples | you run |
|---|---|---|
| raw arrays | `.npy` + numpy | nothing |
| in-process ANN | FAISS, hnswlib, Annoy | nothing — a library |
| embedded DB | LanceDB, Chroma, DuckDB-VSS, sqlite-vec | nothing — a file |
| inside your existing DB | **pgvector** on Postgres | nothing new |
| self-hosted server | Qdrant, Weaviate, Milvus, OpenSearch | a service |
| managed | Pinecone, Qdrant Cloud, OpenSearch Serverless | a bill |

On AWS specifically, the usual paths are **pgvector on RDS/Aurora** and
**OpenSearch Serverless** (which is what Bedrock Knowledge Bases uses underneath
by default). This space moves fast — check what is current at the time rather
than trusting a note.

**Default advice: if the client already runs Postgres, use pgvector.** It is the
lowest marginal complexity in existence — no new system to secure, back up,
monitor or pay for — and it handles the first row and most of the second. Move
to a dedicated vector database when you have a measured reason, not a
speculative one.

---

## 5. The measurement trap — why we are NOT using one yet

**ANN search is approximate. It loses recall on purpose.**

If you measure recall@k on an ANN index, the number you get is chunking quality
+ embedding quality + *ANN recall loss*, mixed together and not separable. You
then cannot tell whether a chunking change helped, because the ANN layer is
adding its own noise.

Our whole experiment is "does the Docling parse beat flat text". Exact brute
force is therefore not a shortcut — it is **the correct instrument**, because it
measures only what we are testing.

The professional order is:

1. Measure with exact search. That is your ceiling and your baseline.
2. Then add ANN and measure what it costs you. Typical HNSW settings give up
   1–5% recall for a large speedup — but *measure it*, do not assume.
3. Choose the operating point knowingly.

Doing it the other way round means you never learn what your retrieval was
actually capable of.

---

## 6. The architectural point worth keeping

**A vector database is a derived artifact, not your source of truth.**

    chunks/<arm>.jsonl.gz      source of truth: the exact text
    vectors/<arm>/<model>/     source of truth: the vectors + meta.json
    <vector database>          a rebuildable index over the above

Keep the raw vectors in object storage even after you adopt a database. Then
re-indexing, switching vendors, or recovering from a corrupted index costs an
import — not a re-embedding run and not a re-parse.

This is the same rule as storing chunk text separately from vectors, one level
up. Each layer should be rebuildable from the one below without redoing the
expensive step.

---

## 7. For this project

Staying with `.npy` + exact matmul through Phase 7 (recall@k) and Phase 8
(hybrid search), because:

- 38 ms per query at our size
- exactness is required to measure the parser comparison cleanly
- BM25 we write ourselves, so fusion happens in our code either way

A vector database becomes worth discussing at Phase 10, if this ever becomes
something served to users rather than measured.
