# When the corpus is not frozen — adding, changing and deleting documents

Written 2026-10-30.

**Caveat before you rely on this.** The other embedding notes are backed by bugs
we watched happen. This one is not — our corpus is fixed and we never updated an
index. Treat it as the list of questions to ask on a client project, not as
proven practice.

This is the single biggest difference between a demo and paid work. Every real
corpus changes. Ours never does, so nothing we built handles it.

---

## 1. Ask this in the first client meeting

- How many documents today, and how many arrive per month?
- Do existing documents get **revised**, or only added to?
- Does anything ever have to be **deleted** — and is that a legal requirement?
- How fresh must the index be: minutes, hours, or "next morning is fine"?

The last one decides the whole architecture. "Next morning" lets you rebuild
everything on a schedule and skip this entire problem.

---

## 2. Price the rebuild BEFORE you engineer incremental updates

This is the step people skip, and it often ends the discussion.

Our full corpus — 196 filings, 137,138 chunks, 43M tokens — costs **$0.80** to
re-embed from scratch, about an hour of one GPU. A nightly full rebuild would be
roughly **$24/month**.

Engineering correct incremental indexing — change detection, partial rewrites,
deletion, and the tests to prove none of it corrupts the index — is days of
work. At $24/month, a full rebuild is cheaper than the engineering for several
years.

**Full rebuild is a legitimate production design, not a shortcut.** It is also
far easier to reason about: there is one code path, and it is the one you
already tested. Incremental indexing earns its complexity only when the rebuild
is genuinely too slow or too expensive — a large corpus, or a freshness
requirement measured in minutes.

Rough shape of where the line sits:

| corpus | full rebuild | verdict |
|---|---|---|
| ~100k chunks | ~1 hour, ~$1 | rebuild nightly, do not build incremental |
| ~1M chunks | ~8 hours, ~$8 | rebuild weekly, incremental for daily additions |
| ~10M+ chunks | a day or more | incremental is required |

---

## 3. What breaks in OUR design if documents change

Our vectors have **no ids**. Row *i* of the concatenated shards is chunk *i* of
`chunks/<arm>.jsonl.gz`, and that positional link is the only thing connecting a
vector to its text.

That is fine for a frozen corpus and fails immediately on an update:

- **Add a document** — append its chunks at the end, append new shards. This one
  is safe, because existing positions do not move.
- **Change a document** — its chunks change. Re-chunking the corpus renumbers
  everything after it, so every vector after that point now points at the wrong
  text. **Nothing errors.** Retrieval silently returns the wrong passages.
- **Delete a document** — same problem, plus you must physically remove rows,
  which means rewriting the shards that contained them.

So the design is append-only by accident, not by intent.

---

## 4. The three fixes, cheapest first

### a. Shard per document instead of per fixed count

Make the shard boundary follow the document, not a count: `shard-<doc_id>.npy`.
Then re-embedding one changed document rewrites exactly one file, and deleting
one deletes one file. You lose uniform shard sizes, which is a small price.

**If I were starting this project again for a client whose documents change,
this is what I would do.** It is a one-line change to the sharding rule and it
removes most of the problem.

### b. Store ids alongside the vectors

Keep an `ids.npy` or `ids.json` parallel to each shard, so a vector carries its
chunk id rather than inheriting it from position. Then order stops mattering and
compaction becomes safe.

Cost: one more file to keep in sync — and a file that can disagree with its
vectors is a new failure mode. Worth it only once positions actually move.

### c. Tombstones plus periodic compaction

Never rewrite on delete. Mark the row dead in a mask (we already have the mask
mechanism for deduplication), filter it at query time, and physically compact on
a schedule.

This is what vector databases do internally, and it is the right answer at scale.
Note that a deleted-but-not-compacted vector is **still on disk** — which matters
for the legal case in `embedding-privacy.md`.

---

## 5. Detecting what actually changed

Do not trust filesystem timestamps or "last modified" from a client's system.
Hash the content:

    doc_hash   = sha256(raw file bytes)      -> did the source change at all
    chunk_hash = sha256(chunk text)          -> did THIS chunk change

Store the hashes with the index. On each run, re-embed only chunks whose hash is
new. A document can be "modified" with no textual change — re-saved, re-exported,
re-OCR'd — and content hashing is what stops you paying to re-embed it.

Second use: the same hash is your deduplication key. We found 12.2% duplicate
chunks in one arm this way.

---

## 6. The part clients forget to tell you

**Deletion is usually a legal requirement, not a feature request.** If a client
operates under GDPR or a similar regime, "delete this person's data" includes
the vectors derived from it — see `embedding-privacy.md` for why vectors count
as derived personal data.

Ask about it explicitly and in writing. Discovering it after you have shipped an
append-only index is an expensive conversation.

---

## 7. What this changes about the vector-database decision

In `vector-stores.md` the argument was: brute-force `.npy` is fine under ~1M
vectors, so do not adopt a database before you need one.

Mutability changes that argument. Add, update and delete are **exactly** what a
vector database gives you for free, and re-implementing them correctly over flat
files is real work. So the decision is not just about scale:

> Adopt a vector database when the corpus is **large**, or when it is
> **mutable and the rebuild is too slow**.

A small, frequently-changing corpus can justify one long before a large frozen
one does.
