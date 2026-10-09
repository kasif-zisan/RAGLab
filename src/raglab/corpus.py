"""Load a staged corpus into LangChain Documents.

The document list comes from data/<dataset>/manifest.json, never from globbing
the directory - see scripts/stage_corpus.py. Two experiments are only
comparable if they ran over byte-identical corpora.
"""

import json
from pathlib import Path

from langchain_core.documents import Document

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
