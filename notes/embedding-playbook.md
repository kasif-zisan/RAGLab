# Embedding — what we did, and what to watch for in client work

## Part 1 — What we did

### Model selection

Four candidates, all accepting at least 8192 input tokens, because our longest
chunks run ~800 tokens and a 512-token model would have truncated a third of
every one of them without raising anything.

Benchmarked on **our real chunks, on the actual T4**, not on synthetic text:

| model | tok/s (short chunks) | tok/s (long chunks) | dims | result |
|---|---|---|---|---|
| gte-modernbert-base | 25,564 | 19,291 | 768 | works |
| Qwen3-Embedding-0.6B | 6,982 | 5,550 | 1024 | works, **chosen** |
| snowflake-arctic-embed-m-v2.0 | — | — | — | needs xformers |
| nomic-embed-text-v1.5 | — | — | — | breaks on transformers v5 |

Two failures, and they were the two models requiring `trust_remote_code=True`.
That flag means the model ships its own Python, written against one generation
of the library; when the library moves, your index becomes unbuildable. **A
natively supported model survives upgrades.**

Chose Qwen3-Embedding-0.6B: 600M parameters against gte-modernbert's 149M.
Cost of that choice was 121 minutes instead of 33, and $0.68 instead of $0.18.

**Note the reasoning that was WRONG:** "1024 dims beats 768 dims". Dimension
count is an architectural property, not a quality score, and is only comparable
*within* one model — the Titan 1024/512/256 case, where AWS measured 512
retaining 99.0%. Across two unrelated models it means nothing, and more
dimensions costs ~33% more arithmetic on every query forever.

### Settings, pinned in one place

    model       Qwen/Qwen3-Embedding-0.6B
    dims        1024
    normalize   true      -> unit vectors, so cosine and dot product rank
                             identically and search is one matrix multiply
    dtype       float16   -> 281 MB instead of 562 MB, ranking unaffected
    doc prefix  ""
    query prefix "Instruct: Given a web search query, retrieve relevant
                 passages that answer the query\nQuery:"

The query instruction came from the model's own
`config_sentence_transformers.json`, **not** from memory or a blog post. Our
first version of the table had it as empty, which would have produced a clean
index, clean queries, no error anywhere, and worse recall.

That instruction applies only to queries, so it can be re-tuned later **without
re-embedding the corpus**. The expensive side is fixed once; the cheap side
stays adjustable.

### How the run was built

    chunks/<arm>.jsonl.gz              the exact text embedded, line = index
    vectors/<arm>/<model>/
        shard-NNNNN.npy                output AND checkpoint
        meta.json                      the contract for query time
    errors/embed/<arm>/<model>/        failures, a DIFFERENT prefix
    bench/candidates.json              measured throughput

Rules carried over from the parse job:

- **The shard file is the checkpoint.** It exists = that range is done. Uploaded
  only after its rows are complete. No state file, no lock.
- **Failures go to a different prefix**, or they read as "done" and are never
  retried.
- **Re-running is always safe**, which is what makes a spot interruption a
  non-event — you lose the shard in flight, nothing else.
- **No partial-shard flush on interruption.** A partial shard is a file whose
  length disagrees with its name: a class of bug not worth saving 60 seconds.

Ran on g4dn.xlarge **spot** at $0.34/hr, with a poller inside the instance
watching `169.254.169.254/latest/meta-data/spot/instance-action`. That notice is
delivered to the instance and is invisible from outside — no external watcher,
console or alarm can see the two-minute warning in time to act.

Ran **two boxes in parallel**, one arm each, in different availability zones.
Two boxes for one hour costs the same as one box for two hours; it just finishes
sooner. Different AZs because a spot reclaim is a property of one capacity pool
= one AZ + one instance type.

### Verification before trusting the output

    row counts proved from S3 OBJECT SIZES    rows*1024*2 + 128 header
      B1  19 shards  37,873 rows  = expected
      A   50 shards  99,265 rows  = expected
    norms   min 0.9998  max 1.0005          -> unit length as configured
    NaN 0,  all-zero rows 0
    self-similarity 1.0000

Row counts came from object sizes alone — no download, no egress charge.

### Result

137,138 vectors, 69 shards, 0 errors, ~1 hour wall clock, **~$0.80**.

### What verification turned up

Arm A's index is **12.2% duplicate text** (`<!-- image -->` placeholders,
repeated running headers) against 0.3% for arm B1. Found by noticing two
vectors in one shard with cosine similarity 1.0.

Containment scoring structurally cannot see this — it asks whether the evidence
is inside *some* chunk, and 12,000 junk chunks do not remove evidence from the
chunks that hold it. recall@k can see it, because a query now has to outrank
12,000 near-identical boilerplate vectors. **This is a testable hypothesis for
why the docling arm might lose, and it only became visible after embedding.**

---

## Part 2 — What to watch for in a professional project

In the order each one bites you.

### 1. Measure your chunk token lengths before choosing a model

The input limit is a hard gate, and over-long input is **truncated, not
rejected**. A 512-token model on 800-token chunks silently discards a third of
every chunk, and you find out as poor recall weeks later.

**Test:** embed your longest chunk, then that chunk plus one sentence. If the
two vectors come back identical, you are over the limit.

### 2. Benchmark on your real data, on the target hardware

Not synthetic text, not a sample from the easy half of your corpus. We first
sampled by *file size* and picked the arm with 2.6x more chunks but half the
length — and overstated throughput by 25%.

**Attention cost grows faster than linearly in sequence length.** A rate
measured on short inputs does not transfer to long ones.

### 3. Leaderboards build a shortlist. Your own test set picks the winner.

MTEB is the right way to get from fifty models to five and the wrong way to pick
one, because your client's documents are not in it. If you have question-to-
evidence labels, you can rank models on the actual corpus — which is the answer
a client is paying for.

### 4. Dimensions are not a quality ranking across models

Only comparable within one model. More dimensions costs storage and ~N% more
arithmetic on every query forever. Pick on measured retrieval, not on a bigger
number.

### 5. Read the prefix/`input_type` convention out of the model's own config

There is no universal convention and it is not transferable between models:

    nomic-embed-text-v1.5   "search_document: " / "search_query: "
    arctic-embed-m-v2.0     ""                  / "query: "
    Qwen3-Embedding-0.6B    ""                  / "Instruct: ...\nQuery:"
    gte-modernbert-base     ""                  / ""

Copy the string exactly, including trailing spaces — it changes tokenisation.
Several models set `default_prompt_name: null`, meaning the library applies
**nothing** unless you ask.

Getting this wrong at query time only raises no error. It just lowers recall.

### 6. Prefer natively supported models over `trust_remote_code=True`

A model shipping its own code is pinned to a library generation and becomes a
dependency you maintain. Both of our failures were exactly these.

It is also real code execution from a third party — on a client's
infrastructure that is a conversation you have deliberately, not a flag you set
quietly.

### 7. Store the chunk text separately from the vectors

Keyed by ID, archived before the run. Then changing the embedding model costs
one embedding run, not a re-parse. If you store only vectors, you have
accidentally made the model choice permanent.

### 8. The output file is the checkpoint

Shard the output. Write each shard only after it completes. Put failures under a
different prefix. Then a re-run resumes, and interruption costs one shard.

This is what makes spot instances correct for this work — and spot is roughly a
third of on-demand. Batch embedding is interruptible; serving is not.

### 9. `meta.json` is the contract between index time and query time

Record model id, dims, normalize, dtype, and the exact prefixes. **Query time
must read its prefix from this file, not from source code**, because a mismatch
between the two produces no error. Write it once — we had a second instance
overwrite it mid-run and lose the git sha.

Include the git sha of the code that produced the vectors. Resolve it on the
machine that *has* the repo; ours resolved it on the instance, which had no
repo, and silently wrote null.

### 10. Verify the vectors before trusting them

- Row counts match chunk counts (provable from object sizes, no download)
- Norms ≈ 1.0 if you asked for normalization
- No NaN, no all-zero rows
- Sample the similarity matrix — near-1.0 off-diagonal pairs mean duplicates

A duplicate rate above a few percent is a retrieval problem, not a storage one.

### 11. Cost shape: per token, or per hour

Managed API = per token: indexing is one-time, queries cost forever.
Your own instance = per hour, whether or not anyone asks a question.

Run the break-even before arguing for self-hosting. A small GPU is ~$730/month;
at ~$0.02 per million tokens that is tens of billions of tokens before the
instance is cheaper. **Self-hosting is about residency, latency or model access
— almost never about cost.**

And for a fixed parallelisable job, more machines costs the same total and
finishes sooner. Parallelise unless something says not to.

### 12. The failures that cost you return a plausible value, not an error

Every bug in this phase did this:

    $None/hr                      a price query missing --start-time
    ap-northeast-11d              a string concatenation off by one
    "git_sha": null               resolved on a machine with no repo
    query prefix ""               taken from memory instead of the config
    "[12m]"                       a loop counter printed as minutes
    a finished log shown live     a watcher reading the wrong file

None of them crashed. A crash tells you where to look; a plausible wrong value
gets written into your results and found weeks later, if at all.

**The defence is checking against something independent.** Row counts against
expected chunk counts. Byte sizes against `rows x dims x 2 + 128`. recall@k
against the containment ceiling. Never a single number on its own.

---

## The one-line version

Embedding is cheap and nearly all of its failure modes are silent, so spend your
effort on **what the model accepts, what prefixes it needs, what the vectors
look like afterwards, and what your numbers are checked against** — not on
picking the model.
