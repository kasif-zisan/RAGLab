# Sharding — what it means, and what ours actually does

Written 2026-10-30, from Phase 7 (embedding 137,138 chunks to vectors).

---

## 1. The word means two different things

Be careful with it in front of a client.

**Database sharding** splits rows of one logical table across *several machines*,
so each holds a subset. The purpose is to scale past what one server can store
or serve. That is the meaning most engineers reach for first.

**Output sharding** — what we do — splits one logical output into *several
files*, all in the same place. Nothing is distributed. The purpose is
checkpointing and resumability, not scale.

Same word, different problem. If you say "we sharded the index" people may hear
a distributed database. Say "the output is written in fixed-size pieces" when
you mean ours.

---

## 2. What ours does

One arm's vectors are not one file. They are N files, each covering a fixed,
contiguous range of chunk positions.

    B1_3200_ov889:  37,873 chunks,  SHARD = 2000  ->  19 files

    shard-00000.npy   chunks     0 .. 1,999     2,000 rows   4,096,128 bytes
    shard-00001.npy   chunks 2,000 .. 3,999     2,000 rows   4,096,128 bytes
    ...
    shard-00018.npy   chunks 36,000 .. 37,872   1,873 rows   3,836,032 bytes
                                                ^^^^^ last one is short

The mapping is arithmetic, not a lookup:

    lo, hi = s * SHARD, min((s + 1) * SHARD, n)

What each name is:

| name | meaning | value for B1 |
|---|---|---|
| `s` | the shard number being written, the loop variable in `for s in range(nshard)` | 0 .. 18 |
| `n` | how many chunks this arm has in total, i.e. `len(texts)` | 37,873 |
| `SHARD` | how many chunks go in one shard — a constant we chose | 2000 |
| `nshard` | how many shards that makes, `(n + SHARD - 1) // SHARD` | 19 |
| `lo`, `hi` | the chunk range this shard covers, half-open: `lo` included, `hi` excluded | e.g. s=3 -> 6000, 8000 |

Worked through for `s = 3`:

    lo = 3 * 2000            = 6000
    hi = min(4 * 2000, 37873) = min(8000, 37873) = 8000
    -> shard-00003.npy holds chunks 6000 .. 7999, which is 2000 rows

And for the last shard, `s = 18`, where `min()` is doing the real work:

    lo = 18 * 2000            = 36000
    hi = min(19 * 2000, 37873) = min(38000, 37873) = 37873
    -> shard-00018.npy holds chunks 36000 .. 37872, which is 1873 rows

**`min(..., n)` is the only thing stopping the last shard reading past the end
of the list.** Without it `texts[36000:38000]` would silently return 1,873 items
anyway — Python slices do not complain — but `hi` would be wrong, and the
`assert v.shape == (hi - lo, dims)` check in the loop would then fire. That
assert exists exactly to catch an off-by-one here.

So shard number alone tells you exactly which chunks are inside. There is no
index file, no manifest, nothing to keep in sync.

### Identity is POSITION, not an ID

This is the part that is easy to miss. The `.npy` files contain **only numbers**
— no chunk ids, no source filenames, no page numbers.

    chunks/<arm>.jsonl.gz   line i   =   chunk i
    shards concatenated     row  i   =   chunk i

Row 4,312 of the concatenated matrix is the chunk on line 4,312 of the jsonl.
That is the entire linkage.

It is why `chunks/<arm>.jsonl.gz` is **frozen and archived before the run**. If
that file is ever regenerated with different chunk boundaries, every vector
silently points at the wrong text. Nothing errors. Retrieval just returns
nonsense that looks plausible.

### Existence is the checkpoint

    def done_shards():
        out = sh(f"aws s3 ls {DEST}/ ...")
        return {int(w.split("shard-")[1].split(".")[0]) ...}

A re-run lists what is already in S3 and skips those numbers. There is no state
file, no database, no lock. **The output IS the progress record.**

Three rules make that safe:

1. **Upload only AFTER the rows are complete.** Upload first and an interrupted
   shard claims "done" for half-finished work, which a re-run then skips
   forever, without error.
2. **Failures go to `errors/`, a different prefix.** A failure written under
   `vectors/` would read as done and never be retried.
3. **S3 `PutObject` is atomic** — a reader never sees a partial object. On a
   local filesystem this does not hold: a half-written file looks exactly like a
   complete one, so there you need write-to-temp-then-rename.

---

## 3. Why 2000, and not 1 or 100,000

Shard size trades recovery granularity against per-object overhead.

| SHARD | files | lost on interruption | notes |
|---|---|---|---|
| 1 | 137,138 | nothing | **$0.64 in S3 PUT requests alone** — comparable to the $0.80 of GPU time the whole run cost |
| 2000 | 69 | ~1–4 min of GPU | what we used; 69 objects cost $0.0003 |
| 100,000 | 2 | up to ~60 min | an interruption near the end loses almost everything |

The sizing question is: **how much work are you willing to redo?** We run on
spot instances with a two-minute interruption warning, so the answer is "a few
minutes". 2000 chunks is 1–4 minutes of GPU and ~4 MB per file.

Memory is *not* the reason here — 99,265 x 1024 fp16 is only 203 MB and would
fit in RAM easily. At 10M chunks it would be 20 GB and memory would become the
driver too.

### Why not just write one file at the end?

- A spot reclaim at minute 59 of 60 loses the entire run.
- There is no resume — a failure restarts from zero.
- You cannot see progress. 69 objects appearing is a progress bar.

### Why not append to one file?

S3 objects are not appendable. You would have to re-upload the whole growing
object each time, which gets quadratically worse.

### Why no partial-shard flush when interrupted?

We considered writing the in-flight rows on the two-minute notice. Rejected: a
partial shard is a file whose length disagrees with its name, and every reader
and resume path then has to handle that case. Not worth it to save ~60 seconds.
The poller stops cleanly instead and the next run redoes that one shard.

---

## 4. What breaks

**Changing `SHARD` between runs silently corrupts the output.** Shard 5 at
SHARD=2000 covers chunks 10,000–11,999. At SHARD=1000 it covers 5,000–5,999. A
resume sees `shard-00005.npy` exists, skips it, and leaves chunks 5,000–5,999
unembedded while believing they are done — and the file that is there holds
completely different rows. No error at any point.

`meta.json` records `shard_size` precisely so this is detectable, and `embed.py`
now asserts the stored value matches before resuming.

**Regenerating the chunks file** breaks the position-to-row linkage described
above. Freeze it.

**Two processes writing the same shards** is wasteful but *not* corrupting here,
because the computation is deterministic — same model, same input, same output
bytes, and PutObject is atomic. We had this happen when two instances both
worked arm A for four minutes. Harmless, but it did overwrite `meta.json`, which
is why that file is now written only if absent.

---

## 5. Reading them back

    import numpy as np, json
    meta = json.load(open("meta.json"))
    v = np.concatenate([np.load(f"shard-{s:05d}.npy")
                        for s in range(meta["shards"])])
    assert v.shape == (meta["chunks"], meta["dims"])

In order, always. `sorted()` on the filenames works because the numbers are
zero-padded to five digits — `shard-00009` sorts before `shard-00010`.
Unpadded, `shard-9` would sort after `shard-10` and the whole matrix would be
silently scrambled. **That is what the `%05d` is for.**
