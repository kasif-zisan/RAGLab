"""Count the creator field - the system that actually laid out each page.

    uv run python scripts/probe_creators.py

producer is the converter that wrote the bytes; creator is the application that
decided where the columns go. If a few creators cover most of the corpus, a
hand-tuned Tier 1 config per creator is affordable. If not, we escalate.
"""

from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "finqa" / "pdf"

# Collapse version noise: "EDGAR Online PDFprocessor V2.0" and "...V3.1" are the
# same layout engine. Match on the vendor, not the version string.
VENDORS = [
    "EDGAR Online", "Merrill", "WebFilings", "Workiva", "RR Donnelley", "Donnelley",
    "Bowne", "Toppan", "Vintage", "Thomson", "DFIN", "Broadridge",
    "QuarkXPress", "FrameMaker", "InDesign", "PageMaker", "Word", "Excel",
    "LaTeX", "Distiller", "Acrobat", "Quartz", "Crystal",
]


def vendor_of(creator: str) -> str:
    for v in VENDORS:
        if v.lower() in creator.lower():
            return v
    return creator[:45] if creator else "<none>"


def main() -> None:
    raw = Counter()
    vendors = Counter()
    by_ticker = defaultdict(set)

    files = sorted(PDF_DIR.glob("*.pdf"))
    for f in files:
        with pymupdf.open(f) as doc:
            md = doc.metadata or {}
        creator = (md.get("creator") or "").strip()
        raw[creator or "<none>"] += 1
        v = vendor_of(creator)
        vendors[v] += 1
        by_ticker[f.stem.split("_")[0]].add(v)

    n = len(files)
    print(f"{n} files, {len(raw)} distinct raw creator strings, "
          f"{len(vendors)} vendors after collapsing\n")

    print("=== by vendor (cumulative coverage) ===")
    run = 0
    for v, c in vendors.most_common():
        run += c
        print(f"{c:4d}  {100*c/n:5.1f}%   cum {100*run/n:5.1f}%   {v}")

    mixed = {t: p for t, p in by_ticker.items() if len(p) > 1}
    print(f"\ncompanies: {len(by_ticker)}  single-vendor: {len(by_ticker)-len(mixed)}"
          f"  mixed: {len(mixed)}")

    print("\n=== raw creator strings, top 15 ===")
    for s, c in raw.most_common(15):
        print(f"{c:4d}  {s[:68]}")


if __name__ == "__main__":
    main()
