"""Arm B: flat PyMuPDF text, re-extracted WITH page boundaries.

    uv run python scripts/extract_flat_pages.py

The benchmark shipped data/finqa/text_pymupdf/ as one blob per filing, with no
page boundaries. Arm A (docling) is one Document per page, so arm B has to be
too - otherwise the two arms differ in granularity as well as in parser, and a
score difference cannot be attributed to either.

Writes data/finqa/flat/<stem>.json in the SAME shape as data/finqa/parsed/, so
one loader serves both arms and they are structurally identical by construction.

~8.3 ms/page measured, so ~3 minutes for 23,000 pages on one core.
"""

import json
import time
from pathlib import Path

import pymupdf

DATA = Path(__file__).resolve().parents[1] / "data" / "finqa"


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    out = DATA / "flat"
    out.mkdir(exist_ok=True)

    t0, total_pages = time.time(), 0
    for n, stem in enumerate(manifest["corpus_docs"], 1):
        pdf = DATA / "pdf" / f"{stem}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(f"manifest lists {stem!r} but {pdf} is missing")

        with pymupdf.open(pdf) as doc:
            pages = [p.get_text() for p in doc]

        # Same keys as parsed/<stem>.json. page_segments == pages always: we did
        # the extraction ourselves, so page i IS page i - no mapping to resolve.
        (out / f"{stem}.json").write_text(json.dumps({
            "stem": stem,
            "pages": len(pages),
            "page_segments": len(pages),
            "tables": 0,              # flat text has no notion of a table
            "chars": sum(len(p) for p in pages),
            "page_markdown": pages,   # same key so one loader reads both arms
            "extractor": f"pymupdf {pymupdf.__doc__.split()[1] if pymupdf.__doc__ else '?'}",
        }))
        total_pages += len(pages)
        if n % 50 == 0 or n == len(manifest["corpus_docs"]):
            print(f"  [{n}/{len(manifest['corpus_docs'])}] {total_pages:,} pages "
                  f"in {time.time()-t0:.0f}s", flush=True)

    el = time.time() - t0
    print(f"\n{total_pages:,} pages -> {out} in {el:.0f}s ({el/total_pages*1000:.1f} ms/page)")


if __name__ == "__main__":
    main()
