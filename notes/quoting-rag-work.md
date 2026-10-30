# Quoting a RAG project — what to price, and what to refuse to price yet

Written 2026-10-30, using this project's real costs.

**Caveat.** The cost figures below are real and measured. The pricing advice is
not — I have not sold a RAG engagement. Treat the numbers as a worked example
and the structure as a starting point to argue with.

---

## 1. The three costs, and the only one clients notice

| | shape | this project |
|---|---|---|
| **One-time build** | your time + a bit of compute | parsing $2.27, embedding $0.80 |
| **Recurring infrastructure** | per month, whether used or not | storage ~$0.12/mo |
| **Per-query** | forever, scales with usage | LLM generation dominates; retrieval ~free |

Total AWS spend for the whole build to date: **under $4.** 196 PDFs, 23,000
pages, 137,138 chunks embedded twice over, plus all the failed attempts.

Internalise that ratio. **Compute is noise; your time is the entire cost.** A
client who wants to negotiate the infrastructure line is negotiating pennies and
should be redirected to scope.

The number they actually live with is **cost per query**, and that is almost
entirely generation tokens — not embedding, not storage, not retrieval. Quote
that number, not a monthly infrastructure figure.

---

## 2. What you must measure before quoting anything

Refuse to give a fixed price before you have these. Every one of them changed a
decision on this project:

1. **A sample of the real documents** — 20 is enough. Scanned? Tables?
   Multi-column? Mixed language? This decides the parser, and the parser decides
   most of the cost. Our Docling parse took 31 core-hours; a text-only corpus
   would have taken minutes.
2. **Real questions from real users** — 20, not invented. Without them you are
   building to your own imagination.
3. **Whether ground truth exists.** Does anyone know the correct answers? If
   not, writing 50-100 question-to-evidence pairs is **the most valuable
   deliverable in the project** and must be its own priced line item. Without it
   you cannot prove anything improved, and neither can they.
4. **Corpus size today and growth rate** — see `index-updates.md`.
5. **Does the data leave their network** — see `embedding-privacy.md`. This can
   eliminate the cheap options entirely.

A client who will not give you documents or questions is a client who cannot be
quoted. Offer a small paid discovery phase instead.

---

## 3. Price it in phases, not as one number

RAG projects go wrong when sold as a single fixed price, because the hard part
is unknowable until you have seen the documents.

A structure that matches how the work actually goes
(see `rag-project-playbook.md`):

| phase | deliverable | fixed-priceable? |
|---|---|---|
| 0 — discovery | document audit, 20 real questions, feasibility | yes, small |
| 1 — ground truth | 50-100 question-to-evidence pairs | yes |
| 2 — parse + chunk | a measured ceiling: "the evidence survives in X% of cases" | yes |
| 3 — retrieval | recall@k, measured | yes |
| 4 — generation | answer quality, measured | **be careful** |
| 5 — production | updates, monitoring, permissions, SLA | separate contract |

Phase 4 is where fixed prices die. Answer quality depends on the documents, the
questions and the model — all largely outside your control. Price the
*measurement* and the *iterations*, not an accuracy target, unless you have
already seen phase 3's numbers.

**Never agree to an accuracy number before phase 2.** The containment ceiling
from phase 2 is the honest upper bound on what is even possible, and it is cheap
to measure. On this project it was 91-95% depending on the arm — had we promised
98% up front, we would have been promising something the parse could not deliver.

---

## 4. Scope boundaries worth writing down

Things clients assume are included and that are each real work:

- **Citations.** "Show me where this came from" requires page-level provenance
  kept through parsing and chunking, and it is unrecoverable if you did not
  plan for it. We nearly lost it and had to redo 35 documents.
- **Permissions.** Per-user filtering at retrieval time.
- **Updates.** New and changed documents. Append-only is not automatic.
- **Deletion.** Often a legal requirement, not a feature.
- **Multiple languages.** A different model, different token economics, a
  different chunk size.
- **A UI.** Clients often mean "a working chat interface" by "a RAG system".
- **Evaluation after you leave.** If they cannot re-run your measurements, every
  future change is a guess. Hand over the test set and the scripts, or sell a
  retainer.

---

## 5. What to show instead of promises

The strongest thing you can put in a proposal is a **measurement from their own
data**, which is why phase 0 and 1 exist. "On your 20 questions, the evidence
survives chunking 94% of the time, so 94% is our ceiling before any retrieval
work" is worth more than any claim about your experience.

The second strongest is a **decision log**. This project has 392 entries, each
one a choice and its reason. When a client asks in month six why the chunk size
is 3200, the answer is a measurement and a date, not a recollection.

That is what this whole `notes/` directory is for, and it is as much a sales
asset as an engineering one.

---

## 6. Red flags in a RAG brief

- "We have millions of documents" — ask to see ten. The answer is usually
  thousands, and the ten will be messier than described.
- "It just needs to answer questions about our data" — no evaluation criteria,
  which means no way to be done.
- "Like ChatGPT but for our documents" — expectations set by a product with a
  very large budget.
- "We'll give you the documents after signing" — you are quoting blind.
- No named person who can say an answer is correct — nobody can sign off.
- "Can you also fine-tune it?" — usually means "we don't know what RAG is for".
  Worth a conversation, not a line item.
