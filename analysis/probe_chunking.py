"""Show what a given chunk_size does to a financial table before we commit to one.

    uv run python scripts/probe_chunking.py                      # 400 / 0
    uv run python scripts/probe_chunking.py --size 1000 --overlap 200
"""

import argparse
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from raglab.corpus import load_corpus

ANCHOR = "Total current liabilities"
HAS_LABEL = re.compile(r"[A-Za-z]{3,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--doc", default="ORLY_2015")
    args = ap.parse_args()

    doc = next(d for d in load_corpus() if d.metadata["source"] == args.doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.size,
        chunk_overlap=args.overlap,
        length_function=len,          # characters, NOT tokens - see (b)
    )
    chunks = splitter.split_documents([doc])

    # Damage metric: a chunk with no word in it is pure numeric soup - it can
    # never be retrieved by a question, and tells you nothing if it is.
    orphans = [c for c in chunks if not HAS_LABEL.search(c.page_content)]

    print(f"{args.doc}: {len(doc.page_content):,} chars "
          f"-> {len(chunks):,} chunks  (size={args.size} overlap={args.overlap})")
    print(f"metadata propagated: {chunks[0].metadata}")
    print(f"orphan chunks (no label at all): {len(orphans)}")

    hits = [i for i, c in enumerate(chunks) if ANCHOR in c.page_content]
    for i in hits:
        for j in range(i - 1, i + 2):
            if 0 <= j < len(chunks):
                tag = ">>> " if j == i else "    "
                body = chunks[j].page_content.replace("\n", " | ")
                print(f"\n{tag}chunk {j}  ({len(chunks[j].page_content)} chars)")
                print(f"{tag}{body}")


if __name__ == "__main__":
    main()
