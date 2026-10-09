"""Compare table-extraction strategies on a page with a known-hard table.

    uv run python scripts/probe_tables.py
    uv run python scripts/probe_tables.py --doc AAPL_2007 --anchor "Total net sales"

Tier 1 of the parsing ladder is free, so it gets tuned before anything is paid
for. Scored against the gold markdown in qa.parquet where one exists.
"""

import argparse
from pathlib import Path

import pymupdf

PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "finqa" / "pdf"

# (vertical, horizontal) - how column edges and row edges are found.
STRATEGIES = {
    "lines":      ("lines_strict", "lines"),   # the default: use drawn rules
    "text":       ("text", "text"),            # infer both from alignment
    "mixed":      ("text", "lines"),           # columns by alignment, rows by rules
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="ORLY_2015")
    ap.add_argument("--anchor", default="Total current liabilities")
    args = ap.parse_args()

    doc = pymupdf.open(PDF_DIR / f"{args.doc}.pdf")
    page = doc[next(p.number for p in doc if p.search_for(args.anchor))]
    print(f"{args.doc} page {page.number}\n")

    for name, (vert, horiz) in STRATEGIES.items():
        tables = page.find_tables(
            vertical_strategy=vert, horizontal_strategy=horiz
        ).tables

        # The two failures we are trying to fix, as numbers: how fragmented is
        # it, and did the year header survive anywhere in the output?
        shapes = [f"{t.row_count}x{t.col_count}" for t in tables]
        md = "\n".join(t.to_markdown() for t in tables)
        header = "2015" in md and "2014" in md

        print(f"=== {name:6s} v={vert:12s} h={horiz:6s}")
        print(f"    {len(tables)} table(s), shapes={shapes[:8]}")
        print(f"    year header present: {header}")

        biggest = max(tables, key=lambda t: t.row_count, default=None)
        if biggest is not None:
            print(f"    largest table ({biggest.row_count} rows):")
            for line in biggest.to_markdown().splitlines()[:6]:
                print(f"      {line}")
        print()

    doc.close()


if __name__ == "__main__":
    main()
