"""Group the corpus by the tool that generated each PDF, not by company.

    uv run python scripts/probe_producers.py
    uv run python scripts/probe_producers.py --ticker AAPL

Layout is a property of the generating software. If one tool made most of the
corpus, one hand-tuned Tier 1 config can cover most of it; if there are 30
tools, hand-tuning does not generalise and we escalate instead.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "finqa" / "pdf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None, help="restrict to one company")
    args = ap.parse_args()

    files = sorted(PDF_DIR.glob("*.pdf"))
    if args.ticker:
        files = [f for f in files if f.stem.startswith(args.ticker + "_")]

    producers = Counter()
    by_ticker = defaultdict(set)

    for f in files:
        with pymupdf.open(f) as doc:
            md = doc.metadata or {}
        # producer = the library that wrote the PDF bytes; creator = the app the
        # author used. producer is the one that controls table geometry.
        prod = (md.get("producer") or "<none>").strip()
        producers[prod] += 1
        by_ticker[f.stem.split("_")[0]].add(prod)

    print(f"{len(files)} files, {len(producers)} distinct producers\n")
    for prod, n in producers.most_common():
        print(f"{n:4d}  {prod[:70]}")

    # Zisan's hypothesis, stated as a measurement: does one company use one tool?
    mixed = {t: p for t, p in by_ticker.items() if len(p) > 1}
    print(f"\ncompanies: {len(by_ticker)}  "
          f"single-producer: {len(by_ticker) - len(mixed)}  mixed: {len(mixed)}")
    for t, p in list(mixed.items())[:5]:
        print(f"  {t}: {len(p)} different producers")


if __name__ == "__main__":
    main()
