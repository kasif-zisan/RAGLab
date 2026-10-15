"""Split page Documents into chunks, per arm.

Each arm gets the separators that suit its own text, because this is a bake-off
between tuned pipelines, not an ablation - see notes/decisions.md 2026-10-11.

    chunk(load_pages("docling"), 1200)
    chunk(load_pages("flat"),     800)
"""

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Docling emits "## Consolidated Balance Sheets" above each table, so markdown
# headers are a real boundary worth splitting on before paragraphs or lines.
MARKDOWN_SEPS = RecursiveCharacterTextSplitter.get_separators_for_language(
    Language.MARKDOWN
)

# Measured, not assumed: the flat PyMuPDF text contains ZERO "\n\n" in
# 78,008,165 characters. The default separator list would spend its first choice
# on something that never occurs, so start where the text actually breaks.
FLAT_SEPS = ["\n", " ", ""]

# flat_pdf is arm B2 - the same text as flat, one Document per filing instead of
# per page, so it gets the same separators.
SEPARATORS = {"docling": MARKDOWN_SEPS, "flat": FLAT_SEPS, "flat_pdf": FLAT_SEPS}


def chunk(
    docs: list[Document],
    chunk_size: int,
    overlap: int | None = None,
) -> list[Document]:
    """Split page Documents, keeping their metadata and numbering the pieces.

    overlap defaults to 15% of chunk_size. Overlap exists to repair evidence cut
    by a boundary - a span that straddles a cut appears whole in one of the two
    chunks instead of half in each. It is not free: it inflates the index by
    roughly its own fraction and puts near-duplicate text in neighbouring
    vectors, so more is not better.
    """
    if not docs:
        return []
    if overlap is None:
        overlap = int(chunk_size * 0.15)

    views = {d.metadata.get("view") for d in docs}
    if len(views) != 1:
        raise ValueError(f"chunk() takes one arm at a time, got views {views}")
    view = views.pop()

    if view == "docling":
        # Markdown tables need structure-aware handling; no LangChain splitter
        # does it. NOTE: for this path chunk_size is a TARGET, not a maximum -
        # 18% of chunks run over at 1200 (p99 3.2x, max 3,899 chars, none near
        # the ~32,000-char embedding limit). The alternative is cutting a table
        # row in half or dropping its header, which defeats the purpose.
        out = []
        for d in docs:
            for piece in split_markdown(d.page_content, chunk_size, overlap):
                out.append(Document(page_content=piece, metadata=dict(d.metadata)))
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=SEPARATORS[view],
            length_function=len,   # CHARACTERS, not tokens - the usual silent misconfig
            keep_separator=True,
        )
        out = splitter.split_documents(docs)

    # Position within the page, so a chunk can be put back in reading order and
    # so neighbours can be fetched for context later.
    seen: dict[tuple, int] = {}
    for c in out:
        key = (c.metadata["source"], c.metadata.get("page_index"))
        c.metadata["chunk_index"] = seen[key] = seen.get(key, -1) + 1
        c.metadata["chunk_size"] = chunk_size
    return out


# ---------------------------------------------------------------------------
# Markdown-table-aware splitting
#
# Measured on a 2,000-page sample (1,488 table blocks, 17,625 rows):
#   100.0%  of table blocks have a separator row
#   100.0%  have it at row 1  -> header is ALWAYS row 0, data from row 2
#    0.07%  contain a second separator (two adjacent tables merged)
#    0.1%   of rows are fully blank
#
# Why this exists: RecursiveCharacterTextSplitter treats a table as text and
# cuts it on "\n". Measured result at chunk_size 1200: 62.3% of chunks holding
# table rows had NO header row - i.e. 26,717 chunks where you cannot tell which
# column is 2015. That is the exact defect docling was bought to fix.
# ---------------------------------------------------------------------------

import re

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_ROW = re.compile(r"^\s*\|.*\|\s*$")
# Must contain a dash. A row of empty cells is also only pipes and spaces, and
# accepting it made blank rows inside a table look like a new table's separator.
_SEP = re.compile(r"^\s*\|[\s:|-]*-[\s:|-]*\|\s*$")


def _table_units(block: list[str], chunk_size: int) -> list[str]:
    """One table block -> units that each fit, every continuation keeping its header.

    A table smaller than chunk_size stays whole: splitting it would be a cut
    with no benefit. Only 41% of tables fit at 1200, but the ones that do are
    the small ones questions are usually about.
    """
    text = "\n".join(block)
    if len(text) <= chunk_size:
        return [text]

    # header is row 0 and separator row 1 in 100% of the corpus; fall back to
    # treating the block as headerless rather than inventing a header.
    head = block[:2] if len(block) > 2 and _SEP.match(block[1]) else []
    body = block[len(head):]
    prefix = "\n".join(head)
    budget = chunk_size - (len(prefix) + 1 if prefix else 0)

    units, cur, cur_len = [], [], 0
    for row in body:
        # A single row wider than the budget still goes out alone - truncating
        # it would destroy the one thing a row chunk is for.
        if cur and cur_len + len(row) + 1 > budget:
            units.append((prefix + "\n" if prefix else "") + "\n".join(cur))
            cur, cur_len = [], 0
        cur.append(row)
        cur_len += len(row) + 1
    if cur:
        units.append((prefix + "\n" if prefix else "") + "\n".join(cur))
    return units


def split_markdown(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split one page of docling markdown, keeping tables readable.

    Tables are atomic units (or header-carrying row groups). Prose is split by
    RecursiveCharacterTextSplitter, which is good at prose. Units are then packed
    greedily, so the sentence introducing a table can share its chunk - that
    context is worth having.

    Each chunk is prefixed with the nearest preceding markdown heading, so
    "Consolidated Balance Sheets" is in the embedded text and in BM25. That is
    the phrase a user's question contains.
    """
    prose_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=MARKDOWN_SEPS, length_function=len, keep_separator=True,
    )

    units: list[tuple[str, str]] = []   # (heading at that point, unit text)
    heading = ""
    buf_rows: list[str] = []
    buf_prose: list[str] = []

    def flush_prose():
        if not buf_prose:
            return
        body = "\n".join(buf_prose).strip()
        if body:
            for piece in prose_splitter.split_text(body):
                units.append((heading, piece))
        buf_prose.clear()

    def flush_rows():
        if len(buf_rows) >= 2:
            for u in _table_units(buf_rows, chunk_size):
                units.append((heading, u))
        elif buf_rows:
            buf_prose.extend(buf_rows)     # a lone pipe line is not a table
        buf_rows.clear()

    for line in text.splitlines():
        if _ROW.match(line):
            flush_prose()
            buf_rows.append(line)
            continue
        flush_rows()
        m = _HEADING.match(line)
        if m:
            flush_prose()
            heading = m.group(2).strip()
        buf_prose.append(line)
    flush_rows()
    flush_prose()

    # Pack units greedily. A unit is never cut here - it was already sized.
    chunks, cur, cur_head, cur_len = [], [], None, 0
    for h, u in units:
        crumb = f"{h}\n\n" if h else ""
        add = len(u) + 1
        if cur and (cur_len + add > chunk_size or h != cur_head):
            chunks.append((f"{cur_head}\n\n" if cur_head else "") + "\n".join(cur))
            cur, cur_len = [], 0
        cur_head = h
        if not cur:
            cur_len = len(crumb)
        cur.append(u)
        cur_len += add
    if cur:
        chunks.append((f"{cur_head}\n\n" if cur_head else "") + "\n".join(cur))
    return [c for c in chunks if c.strip()]
