"""Did the gold evidence survive chunking?

The retrieval ceiling. FinQA ships the exact text a human marked as containing
each answer (`qa.parquet.source_text`). If no single chunk holds a complete gold
span, no embedding model, reranker or prompt can recover it - the answer was
destroyed before retrieval ever ran. Measuring this costs nothing, so it runs on
every sweep point BEFORE we pay to embed anything.

Why this is not a string comparison
-----------------------------------
FinQA's annotations are its own extractor's output, not the filing's text. Five
measured corruptions, each of which breaks `gold in chunk`:

1. Tokenizer spacing - "december 31 , 2006", "$ 25.4 million". Naive normalised
   substring match scored 0 of 139 prose spans.
2. Thousands separators stripped - gold "3547", filing "3,547". Normalising just
   this took prose from 0 to 104 of 139.
3. Words glued across a line break. A header cell laid out as "Average Price" /
   "Paid per Share" became "Average PricePaid per Share". Docling renders the
   space. So the comparison is on CONCATENATED characters, aligned to token
   boundaries, which matches "pricepaid" to "price"+"paid" in either direction.
4. Raw HTML left in the annotations - "Shares<i>(in thousands)</i>",
   "<sup>(a)</sup>". Stripped from gold before tokenising.
5. Mojibake - the benchmark lost its own \\u escapes, so "2013" stands for an en
   dash, "2019" for a right quote, "a031" for a non-breaking space. These tokens
   do not exist in the filing. We do NOT delete them, because in a financial
   table 2013 and 2014 are real column headers; instead prose matching tolerates
   a small fraction of unmatched gold tokens (see PROSE_MIN_FRAC).

Fixes 3 and 4 together took locatability from A 66.7%/B 70.4% to A 83.8%/B 87.6%,
and fix 5 took unlocatable-by-both from 127 spans to 44.

Every one of these repairs changes the NEEDLE, not the haystack, and is applied
identically to both arms - so it cannot favour a parser. That is the test for
whether a repair is legitimate. Rebuilding the gold spans FROM our own parsed
output would fail that test: the arm that generated them would score 100% by
construction.

Why the haystack is one chunk, not one document
-----------------------------------------------
Measured: at document level the metric does not discriminate (arm A 66.7%, arm B
70.4% - flat text AHEAD) because a 126-page filing almost always has somewhere
the tokens line up, often a second copy of the same table. Per chunk the search
space is ~1000 characters, so coincidental matches cannot rescue a destroyed
table. It is also the honest question: a retriever hands the model one chunk.

What it does NOT measure
------------------------
Presence, not answerability. Flat text can hold every cell of a row while having
lost the column header, so "which year is this figure?" is unanswerable even
though the span counts as contained. Report that limit; do not overclaim.
"""

import json
import re
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

# Only commas BETWEEN digits: "3,547" -> "3547" matches gold, while "x, y" is
# left alone so it still tokenises as two words rather than one.
_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
# Bounded length so it cannot swallow a '<' used as less-than across half a page.
_TAG = re.compile(r"<[^>]{1,20}>")
_TOKEN = re.compile(r"[a-z0-9]+")
_SEP_CHARS = set(" -:")

# A gold row's cells must all fall inside this multiple of the row's own token
# count. Validated on 1,166 true gold rows and 1,159 label-swapped fakes: at
# 1.0-1.5 both arms keep ~90% of true rows and reject >99% of fakes. At 3.0
# false positives reach 4.5-6.3% and arm B starts accepting a row whose label
# belongs to a different company - proven on the ORLY performance-graph table,
# which PyMuPDF emitted column-major (all labels, then one run of numbers).
ROW_SLACK = 1.5

# Prose: fraction of gold tokens that must match, in order, inside a window of
# PROSE_STRETCH times the span's length. Absorbs the mojibake of fix 5 without
# opening the door: measured 88.5% recovery at 0% false positives against wrong
# documents. Tables stay strict - the table failures that remain are genuine
# column reordering, which must count as a failure.
PROSE_MIN_FRAC = 0.94
PROSE_STRETCH = 1.3


def tokens(text: str, strip_tags: bool = False) -> list[str]:
    """Canonical token stream: tags out, numbers de-comma'd, lowercase words."""
    if strip_tags:
        text = _TAG.sub(" ", text)
    return _TOKEN.findall(_DIGIT_COMMA.sub("", text.lower()))


class Haystack:
    """Text prepared for glue-insensitive search.

    Holds the tokens, their concatenation, and the character offset where each
    token begins. A cell matches only when its concatenated characters sit at a
    token boundary on both ends - that is what lets "pricepaid" match
    "price"+"paid" without also matching the tail of some unrelated word.
    """

    def __init__(self, text: str):
        self._build(tokens(text))

    @classmethod
    def from_tokens(cls, tok: list[str]) -> "Haystack":
        """Reuse an already-computed token list instead of re-tokenising."""
        self = cls.__new__(cls)
        self._build(tok)
        return self

    def _build(self, tok: list[str]) -> None:
        self.tok = tok
        self.chars = "".join(self.tok)
        offsets, n = [], 0
        for t in self.tok:
            offsets.append(n)
            n += len(t)
        offsets.append(n)
        self.offsets = offsets
        self.index_of = {c: i for i, c in enumerate(offsets)}
        self.where = defaultdict(list)
        for i, t in enumerate(self.tok):
            self.where[t].append(i)

    def find(self, cell: list[str], from_tok: int) -> tuple[int, int] | None:
        """First token-aligned occurrence of the glued cell at/after from_tok."""
        glued = "".join(cell)
        if not glued:
            return None
        at = self.offsets[min(from_tok, len(self.tok))]
        while True:
            k = self.chars.find(glued, at)
            if k < 0:
                return None
            end = k + len(glued)
            if k in self.index_of and end in self.index_of:
                return self.index_of[end], self.index_of[k]
            at = k + 1

    def starts(self, token: str, from_tok: int) -> list[int]:
        pos = self.where.get(token)
        if not pos:
            return []
        return pos[bisect_left(pos, from_tok) :]


def gold_rows(span: str) -> list[list[list[str]]]:
    """A gold markdown table -> rows -> non-empty cells -> tokens."""
    rows = []
    for line in span.splitlines():
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s[1:-1].split("|")]
        if all(set(c) <= _SEP_CHARS for c in cells):
            continue
        cell_tokens = [t for t in (tokens(c, True) for c in cells) if t]
        if cell_tokens:
            rows.append(cell_tokens)
    return rows


def _row_in(row: list[list[str]], hay: Haystack, start: int) -> int | None:
    """Cells of `row` in order at/after `start`, all inside one window."""
    budget = ROW_SLACK * max(4, sum(len(c) for c in row))
    at = start
    while True:
        hit = hay.find(row[0], at)
        if hit is None:
            return None
        pos, begin = hit
        ok = True
        for cell in row[1:]:
            nxt = hay.find(cell, pos)
            if nxt is None or nxt[0] - begin > budget:
                ok = False
                break
            pos = nxt[0]
        if ok:
            return pos
        at = begin + 1


def _glue_step(
    hay: list[str], j: int, want: list[str], k: int, span: int = 3
) -> tuple[int, int] | None:
    """Match want[k:k+b] against hay[j:j+a] by concatenated characters.

    Returns (a, b) for the smallest match, or None. This is what makes prose
    matching glue-insensitive, the same way Haystack.find already is for table
    cells. Without it the scorer PUNISHES a parser for being correct: a filing
    hyphenated "com- pensation" across a line break, docling rejoins it to
    "compensation" (right), PyMuPDF keeps the break (wrong) and so accidentally
    matches the gold's broken token. Measured: 8 prose spans findable in flat
    text but not in docling, which read as a docling defect and were not.
    """
    for total in range(2, 2 * span + 1):
        for a in range(1, min(span, total - 1) + 1):
            b = total - a
            if b < 1 or b > span or j + a > len(hay) or k + b > len(want):
                continue
            if "".join(hay[j : j + a]) == "".join(want[k : k + b]):
                return a, b
    return None


def _prose_in(span: str, hay: Haystack) -> bool:
    """Gold tokens in order inside a bounded window, allowing a few misses.

    Glue-insensitive (see _glue_step), so hyphenation and line-break joins do
    not decide the result.
    """
    want = tokens(span, True)
    n = len(want)
    if n < 4:
        return False
    need = -(-round(PROSE_MIN_FRAC * n * 100) // 100)  # ceil, no float drift
    limit = int(PROSE_STRETCH * n)
    cand = hay.starts(want[0], 0) or hay.starts(want[1], 0)
    toks_hay = hay.tok
    for start in cand:
        matched, j, k = 0, start, 0
        while k < n and j < len(toks_hay) and (j - start) <= limit:
            step = _glue_step(toks_hay, j, want, k)
            if step is not None:
                a, b = step
                matched += b
                k += b
                j += a
                continue
            # gold carries a token the filing does not (the mojibake of fix 5):
            # drop it and try to land the one after. The dropped token is NOT
            # counted as matched, so PROSE_MIN_FRAC still bounds how many the
            # span may lose.
            step = _glue_step(toks_hay, j, want, k + 1) if k + 1 < n else None
            if step is not None:
                a, b = step
                matched += b
                k += 1 + b
                j += a
                continue
            j += 1
        if matched >= need:
            return True
    return False


def span_in(span: str, hay: Haystack) -> bool:
    """Is this complete gold span present in this text?

    Table spans: every gold row matches, in document order, each row's cells
    inside one window. Prose spans: ordered match with bounded slack.
    """
    rows = gold_rows(span)
    if rows:
        pos = 0
        for row in rows:
            nxt = _row_in(row, hay, pos)
            if nxt is None:
                return False
            pos = nxt
        return True
    return _prose_in(span, hay)


def load_gold(dataset: str = "finqa") -> pd.DataFrame:
    """393 questions. `source_text` is an array of 1-4 spans; ALL must survive."""
    return pd.read_parquet(Path("data") / dataset / "qa.parquet")


class _Chunk:
    """A chunk's tokens, with its search index built only when needed.

    Building the full Haystack (character concatenation, offset table, inverted
    index) for every one of ~150k chunks is most of the scorer's cost, and the
    token-set prefilter rejects the overwhelming majority of them. So tokenise
    eagerly - that is cheap and the prefilter needs it - and build the index
    lazily, only for the handful of chunks that could actually hold the span.
    """

    __slots__ = ("doc", "tok", "tokset", "chars", "_hay")

    def __init__(self, doc: Document):
        self.doc = doc
        self.tok = tokens(doc.page_content)
        self.tokset = set(self.tok)
        # Concatenated tokens, for the prose prefilter. A set of token VALUES is
        # the wrong gate in front of a glue-insensitive matcher: gold "com" +
        # "pensation" is matched by the filing's "compensation", but neither
        # token value is in the set, so the chunk was skipped before the matcher
        # ran. Measured: the glue fix changed A_parse by 0.000% until this gate
        # was fixed too.
        self.chars = "".join(self.tok)
        self._hay: Haystack | None = None

    @property
    def hay(self) -> Haystack:
        if self._hay is None:
            self._hay = Haystack.from_tokens(self.tok)
        return self._hay


def score(chunks: list[Document], dataset: str = "finqa") -> pd.DataFrame:
    """One row per (question, span): did any single chunk hold it whole?

    Joins on qa.parquet.file_name == chunk metadata["source"].

    `chunks_touched` counts chunks that share at least one token with the span
    but do not hold it - a rough measure of how badly the span was scattered.
    """
    by_source: dict[str, list[Document]] = {}
    for c in chunks:
        by_source.setdefault(c.metadata["source"], []).append(c)
    prepared = {
        stem: [_Chunk(d) for d in docs] for stem, docs in by_source.items()
    }

    out = []
    for qid, row in load_gold(dataset).iterrows():
        stem = row["file_name"]
        for si, span in enumerate(row["source_text"]):
            rows = gold_rows(span)
            is_table = bool(rows)
            if is_table:
                # A chunk cannot hold a table span unless it holds every token
                # of it, in some order. Sound: the ordered scan only ever needs
                # tokens the set already contains.
                need = {w for r in rows for c in r for w in c}
                want, floor = None, 0
            else:
                # Prose tolerates misses, so "holds every token" is too strong.
                # Sound bound instead: any gold token the matcher DOES match
                # must appear as a substring of the chunk's concatenated tokens
                # (that is what glue matching means), so cap the achievable
                # match count by how many gold tokens appear as substrings.
                # Substring, not set membership - see _Chunk.chars.
                want = tokens(span, True)
                floor = -(-round(PROSE_MIN_FRAC * len(want) * 100) // 100)
                need = None

            found, hit_at, n_partial = False, None, 0
            for ch in prepared.get(stem, ()):
                have = ch.tokset
                if is_table:
                    if not need <= have:
                        if need & have:
                            n_partial += 1
                        continue
                else:
                    text = ch.chars
                    reachable = sum(1 for w in want if w in text)
                    if reachable < floor:
                        if reachable:
                            n_partial += 1
                        continue
                if span_in(span, ch.hay):
                    # `found` must NOT be derived from chunk_index: unchunked
                    # Documents carry no chunk_index and every hit would read as
                    # a miss. Measured the hard way - it printed 0.0%.
                    found = True
                    hit_at = ch.doc.metadata.get("chunk_index")
                    break
            out.append(
                {
                    "qid": qid,
                    "source": stem,
                    "span_index": si,
                    "is_table": is_table,
                    "gold_chars": len(span),
                    "contained": found,
                    "chunk_index": hit_at,
                    "chunks_touched": n_partial,
                }
            )
    return pd.DataFrame(out)
