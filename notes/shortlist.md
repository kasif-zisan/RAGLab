# Phase 7 shortlist — candidates from the free sweeps

Decided 2026-10-26, after 21 chunk-size points and 20 overlap points, all
measured on gold evidence containment (free, no embeddings).

Definitions live in `src/raglab/configs.py`. This file is the reasoning; that
file is the single source of the numbers.

## The three representations

| arm | parser | one Document per | note |
|-----|--------|------------------|------|
| A | Docling markdown | page | table-aware splitter, keeps tables whole |
| B1 | PyMuPDF flat text | page | |
| B2 | PyMuPDF flat text | filing | identical text to B1, no page boundaries |

## Shortlist

| config | chunks | tables | prose | questions | why it is here |
|--------|--------|--------|-------|-----------|----------------|
| **A @2400 ov=360** | 99,265 | 91.0 | 94.7 | 90.7 | the best docling can do — tables are AT its parse ceiling |
| A @1200 ov=542 | 178,051 | 72.9 | 93.1 | 76.5 | small-chunk docling, for the dilution test |
| **B1 @3200 ov=889** | 37,873 | 94.6 | 100.0 | 95.7 | leader on containment, 38% of A's chunk count |
| B1 @1600 ov=889 | 94,354 | 92.8 | 100.0 | 94.2 | small-chunk flat, for the dilution test |
| B2 @2400 ov=889 | 51,579 | 94.6 | 100.0 | 95.7 | per-filing variant, ties B1@3200 |

Bold = selected for the first indexes.

Parse ceilings (best possible, no chunking at all): A tables 91.0 / prose 96.2.
B tables 94.9 / prose 100.0.

## Why these two first

**A @2400 ov=360** is docling's best possible result, so if docling loses here it
loses everywhere. Overlap is irrelevant for this arm — tables 91.0 at every
overlap from 0 to 889 — because the splitter keeps tables whole and 38% of its
chunks are one entire table, leaving nothing partial to carry.

**B1 @3200 ov=889** beats it on every measure at 38% of the chunk count. Overlap
889 is the p99 of gold evidence length, following the rule that overlap at least
as long as the evidence guarantees the evidence is never cut. It costs only
1.38x the index, because large chunks make overlap cheap.

## What these numbers cannot tell us

1. **Containment rises with both chunk size and overlap.** It narrows options and
   cannot pick a winner. Letting it pick selects the largest chunk every time.
2. **Dilution is invisible here.** One chunk becomes exactly 1024 numbers however
   long it is, so a 3200-character chunk covering several topics gets one
   position that represents none of them well. Only recall@k sees this.
3. **Presence is not answerability.** PyMuPDF dumped the ORLY performance table
   column-first: a chunk can hold every number and have lost which company each
   belongs to. Containment scores that as a hit. A model reading it cannot answer.
   This is where docling may still win, and it is only visible at Phase 9.
4. **The uncertainty is about +/- 3 points.** 345 questions, 408 spans. Any
   difference under ~3 points reported today was inside the noise.
5. **The arms agree on 341 of 345 questions.** The whole comparison rests on 44
   disagreements (15 docling-only, 29 flat-only), exact McNemar p = 0.0488 —
   significant by a hair, with the chi-squared approximation landing on the other
   side of the line. Do not lean on it.

## Run order

Two separate vector stores, built one after the other — never one combined
store, which would make a retrieval hit unattributable to a parser. Sequential
also means the first embedding bill is visible before the second is incurred.

A single store holding both representations is the "arm C" dual-view idea. That
is a different experiment and comes after this one resolves.
