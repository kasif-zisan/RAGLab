"""Load a staged corpus into LangChain Documents.

Two representations of the same 196 filings, for the A/B/C experiment:

    load_pages("docling") -> arm A, docling markdown, one Document per PAGE
    load_pages("flat")    -> arm B, PyMuPDF text,     one Document per PAGE
    load_filings("flat")  -> arm B2, same text as B1 but one Document per FILING
    load_corpus()         -> the original shipped text_pymupdf/ blobs, kept for
                             the chunk-size work; NOT an arm of the experiment

Arms A and B must differ in ONE thing - the parser. So both are per page, both
read the same JSON shape, and both go through the same function. If arm B were
per filing it would differ in granularity too, and a score gap could not be
attributed to either cause.

The document list comes from data/<dataset>/manifest.json, never from globbing
the directory - see scripts/stage_corpus.py. Two experiments are only
comparable if they ran over byte-identical corpora.
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document

HAS_WORD = re.compile(r"[A-Za-z]{3,}")

DATA = Path(__file__).resolve().parents[2] / "data"


def load_corpus(dataset: str = "finqa") -> list[Document]:
    root = DATA / dataset
    manifest = json.loads((root / "manifest.json").read_text())
    gold = set(manifest["gold_docs"])

    docs = []
    for stem in manifest["corpus_docs"]:
        path = root / "text_pymupdf" / f"{stem}__pymupdf.txt"
        if not path.exists():
            raise FileNotFoundError(f"manifest lists {stem!r} but {path} is missing")
        text = path.read_text(encoding="utf-8")

        ticker, _, year = stem.partition("_")
        docs.append(Document(
            page_content=text,
            metadata={
                "source": stem,          # answer "where did this come from?"
                "ticker": ticker,        # metadata filtering later
                "year": year,
                "is_gold": stem in gold,
                "chars": len(text),      # so we can see the size problem
            },
        ))

    return docs


VIEWS = {"docling": "parsed", "flat": "flat"}


def load_pages(view: str = "docling", dataset: str = "finqa") -> list[Document]:
    """One Document per PAGE, from either parser.

    Per page, not per filing, because a chunk that spans a page boundary has no
    single page of origin - and then no citation. Splitting inside a page means
    every chunk can name exactly one page. A page here averages ~4,700 chars,
    which is large enough to hold a whole table and small enough that the
    splitter still has work to do.
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {sorted(VIEWS)}, got {view!r}")

    root = DATA / dataset
    manifest = json.loads((root / "manifest.json").read_text())
    gold = set(manifest["gold_docs"])

    docs = []
    for stem in manifest["corpus_docs"]:
        path = root / VIEWS[view] / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(f"manifest lists {stem!r} but {path} is missing")
        rec = json.loads(path.read_text())

        # Docling emits no page-break for some pages, so for it the segment
        # position is the true page number only when the counts agree; the other
        # 36.7% of filings resolve the page from the PDF at citation time (see
        # notes/decisions.md 2026-10-11). For the flat view we did the
        # extraction ourselves, so this is always True.
        trusted = rec["page_segments"] == rec["pages"]

        ticker, _, year = stem.partition("_")
        for i, seg in enumerate(rec["page_markdown"]):
            # A page with no word in it is a scanned logo or a blank: it can
            # never be retrieved by a question, and costs a full-weight vector.
            if not HAS_WORD.search(seg):
                continue
            docs.append(Document(
                page_content=seg,
                metadata={
                    "source": stem,
                    "ticker": ticker,
                    "year": year,
                    "is_gold": stem in gold,
                    "view": view,           # "docling" or "flat" - which arm
                    "page_index": i,        # a hint, exact only if page_trusted
                    "page_trusted": trusted,
                    "chars": len(seg),
                },
            ))

    return docs


def load_filings(view: str = "flat", dataset: str = "finqa") -> list[Document]:
    """Arm B2: the same pages as load_pages(), joined back into one Document.

    Built by concatenating our own flat/ pages rather than reading the shipped
    text_pymupdf/ blobs, which differ by -0.04% (a PyMuPDF version difference).
    Identical text means B1 and B2 differ in ONE thing - whether a chunk may
    cross a page boundary - which is the only question B2 exists to answer.

    The trade this arm tests: chunks may cross page boundaries, so a table split
    across two pages stays contiguous; but no chunk can name a page, so this arm
    cannot produce page-level citations. page_index is None and page_trusted is
    False, deliberately - that is the honest encoding of what it gives up.
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {sorted(VIEWS)}, got {view!r}")

    root = DATA / dataset
    manifest = json.loads((root / "manifest.json").read_text())
    gold = set(manifest["gold_docs"])

    docs = []
    for stem in manifest["corpus_docs"]:
        path = root / VIEWS[view] / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(f"manifest lists {stem!r} but {path} is missing")
        rec = json.loads(path.read_text())
        # "" not "\n": get_text() already ends each page with a newline, so this
        # reproduces the shipped concatenation exactly.
        text = "".join(rec["page_markdown"])

        ticker, _, year = stem.partition("_")
        docs.append(Document(
            page_content=text,
            metadata={
                "source": stem,
                "ticker": ticker,
                "year": year,
                "is_gold": stem in gold,
                "view": f"{view}_pdf",   # own arm: different separators are fine,
                "page_index": None,      # but it can never cite a page
                "page_trusted": False,
                "chars": len(text),
            },
        ))
    return docs
