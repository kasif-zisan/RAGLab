# RAG project playbook — the order to do things in

Written 2026-10-15, from the FinQA build. Plain-language checklist for running a
client RAG project, and for telling Claude what to do next.

The point of the order: **each step produces the thing the next step needs.** Skip
one and you end up guessing, or worse, measuring your own mistake.

---

## Phase 0 — Find out what you are actually building (no code)

- Get 20 real questions the client's users will ask. Not made-up ones.
- Ask: for each question, is there a correct answer somewhere we can check against?
- Ask about money: what is a one-time cost here vs a cost on every question?
  Parsing and building the search index are one-time. Every question a user asks
  costs tokens forever.

**Decide:** is this answerable from documents at all, or do they need a database query?

**Skip this and:** you build something that works on your examples and fails on theirs.

---

## Phase 1 — Open the documents yourself

Open five by hand. Write down what is in them: plain text, tables, scanned
images, forms, multi-column layouts.

**Decide:** which parser to try. Tables and scans need a stronger parser than
plain text, and stronger means slower and sometimes priced per page.

**Skip this and:** you pick a parser by reputation instead of by what your
documents contain.

---

## Phase 2 — Parse, then check the parse

Try the cheapest parser that could survive your document type. Then check it:

- Pick the hardest page you can find and read the output yourself.
- For a table: did it survive, and can you still tell which number belongs to
  which row and column? A parser can output every number and still destroy
  which-number-goes-where. That is worse than failing, because it looks fine.
- Record failures per document, not just a total. One broken document out of 200
  is fine; twenty broken ones in the same format is a pattern.

**Decide:** is this parser good enough, or do you escalate to a more expensive one?

**Skip this and:** every later number is measured on broken input and you will
blame your chunking for a parsing problem.

---

## Phase 3 — Build the test set BEFORE you chunk anything

You need: question -> the exact text in the document that answers it.

- If the client has this, great. If not, you write it for 50-100 questions. This
  is slow and it is the most valuable thing you will produce.
- **Then test your test set.** Run your checking code against the parsed
  documents with no chunking at all. Can it find the answer text? If not, the
  test set or the checking code is wrong, and you must fix that before you judge
  anything else.
- Expect the client's labels to be messy: different spacing, stray markup,
  numbers written differently, text that is not actually in the document.

**Decide:** which questions are usable. Throw out the ones whose answer text
cannot be found anywhere in the document, and say how many you threw out.

**Skip this and:** you have opinions instead of measurements, and you cannot tell
a client whether your changes helped.

---

## Phase 4 — Measure how long the answer text is

Look at your test set and measure the length of the answer text: typical, and
worst case.

**Decide:** your minimum chunk size. If a chunk is smaller than the answer text,
the answer gets cut in half and nothing downstream can fix it.

**Skip this and:** your chunk size is a guess, and when the client asks "why
1200?" you have no answer.

---

## Phase 5 — Chunk

- Split on boundaries that actually exist in your documents. Check this — do not
  assume. (In our flat text there were zero blank-line breaks in 78 million
  characters, so the default splitter's first choice was useless.)
- Keep tables whole if they fit. If they do not, repeat the header row into every
  piece, or the numbers arrive with no idea what they are.
- Keep the document name and page number on every chunk, so you can cite later.

**Skip this and:** you get chunks that start mid-table with no header.

---

## Phase 6 — Free check: did the answer text survive chunking?

For each question: is the complete answer text inside **one single chunk**?

This costs nothing — no embedding, no index, no API calls. It gives you the
**best accuracy that is still possible**. If the answer was cut in half, no
search method, no reranker and no prompt can recover it.

Run it at several chunk sizes.

**Two rules:**

1. This is a ceiling, not a score. 96% here does not mean 96% accuracy. It means
   96% is the most you could possibly get.
2. **It always improves with bigger chunks.** A huge chunk contains everything.
   So this step narrows your options; it never picks for you. If you let it pick,
   it picks the biggest size every time.

**Decide:** a shortlist of 4-6 chunk sizes worth paying to test.

---

## Phase 7 — Paid check: can a retriever actually find it?

Build the search index, then measure: ask the question, take the top k results,
did one of them contain the answer text?

This is where big chunks start to hurt. A big chunk covers many topics, so its
one summary number represents all of them badly and the search engine stops
finding it reliably.

**Decide:** the real chunk size. It is where Phase 6 stops improving and Phase 7
has not yet got worse.

---

## Phase 8 — Improve the search, one change at a time, measuring after each

1. Keyword search and vector search, results combined
2. A reranker over the combined results
3. Measure after each change. If the number did not move, the change did nothing
   and you remove it.

**Skip the measuring and:** you ship four changes, the system is better, and you
cannot tell the client which change was worth the money.

---

## Phase 9 — Check the answers, not just the search

A chunk can contain the right number and still produce a wrong answer, because
the column header saying which year it is got left behind. "The text was
retrieved" and "the question is answerable" are different things.

**Decide:** is the generated answer actually supported by what was retrieved?

---

## Phase 10 — Cost and citations before you call it done

- Cost per question, not per project. That is the number the client lives with.
- Can you show the user the page the answer came from? For finance, legal and
  medical work this is usually mandatory, not a nice extra.

---

## Rules that apply at every single step

1. **After any change meant to improve a number, look at the number.** A passing
   unit test is not evidence the system improved. (This build: a fix passed its
   test and moved the real number by 0.000%, because a filter in front of it
   blocked the fix entirely.)

2. **When one option looks worse, suspect your measurement first.** In this build,
   three of the differences between the two parsers turned out to be faults in my
   own scoring code, not real differences.

3. **Compare options at the same settings.** Never compare option A tuned one way
   against option B tuned another way. A 16-point difference here turned out to
   be ~2 points once both were tested the same way.

4. **Fix the test set, never generate it from your own output.** Cleaning up the
   client's labels is fine. Producing labels from your own pipeline's output means
   your pipeline scores 100% and you have measured nothing. The test: could this
   label have been written without looking at my system's output?

5. **Write down every decision and why, as you go.** You will be asked to justify
   choices months later, and "it measured better" needs the measurement attached.

---

## How to tell Claude what to do

Useful instructions name the phase and demand the check:

- "We are at Phase 2. Parse these 10 documents, then show me the hardest table
  and tell me whether which-number-goes-where survived."
- "Phase 3. Before any chunking, check whether our checking code can find the
  answer text in the unchunked documents. Report how many it cannot find and why."
- "Phase 6. Run the free containment check at 800/1200/1600/2000 and give me a
  shortlist. Do not pick a winner from it."
- "You just changed the matcher. Re-run the controls and tell me whether the
  aggregate number moved. If it did not, say so."
- "Compare these at the same chunk size, not at the sizes you picked for each."
