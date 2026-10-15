"""Does the containment metric get LOOSER as chunks get bigger?

It must, in principle. We proved the metric is inflated at document level: a
126-page filing almost always has somewhere the tokens line up, which is why
arm B scored HIGHER than arm A there despite its tables being destroyed. A chunk
is a small haystack, so the metric is strict - but as chunk_size grows the chunk
approaches the document and the inflation must creep back.

This quantifies it per sweep point, so the sweep table can be read with the
right amount of trust. Control: a label-swapped gold row - every cell present,
the fact wrong. Any chunk accepting it is a false positive.
"""

import argparse

import pandas as pd

from raglab.chunking import chunk
from raglab.corpus import load_filings, load_pages
from raglab.scoring import Haystack, _row_in, gold_rows, load_gold, tokens


def fake_rows() -> list[tuple[str, list[list[str]]]]:
    """One falsified row per real gold row: its label swapped for a sibling's."""
    out = []
    for _, r in load_gold().iterrows():
        for span in r["source_text"]:
            data = [x for x in gold_rows(span) if len(x) >= 4]
            if len(data) < 2:
                continue
            for i, row in enumerate(data):
                other = data[(i + 1) % len(data)]
                if other[0] != row[0]:
                    out.append((r["file_name"], [other[0]] + row[1:]))
    return out


def fp_rate(chunks, fakes) -> float:
    by_source: dict[str, list[Haystack]] = {}
    for c in chunks:
        by_source.setdefault(c.metadata["source"], []).append(c)
    prepared = {
        stem: [(set(t), t) for t in (tokens(d.page_content) for d in docs)]
        for stem, docs in by_source.items()
    }
    bad = 0
    for stem, row in fakes:
        need = {w for cell in row for w in cell}
        for tokset, toklist in prepared.get(stem, ()):
            if not need <= tokset:
                continue
            if _row_in(row, Haystack.from_tokens(toklist), 0) is not None:
                bad += 1
                break
    return bad / len(fakes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[800, 1200, 1600, 2000, 2400, 3200, 4000])
    args = ap.parse_args()

    fakes = fake_rows()
    print(f"falsified gold rows: {len(fakes)}\n", flush=True)
    loaders = {
        "A": lambda: load_pages("docling"),
        "B1": lambda: load_pages("flat"),
        "B2": lambda: load_filings("flat"),
    }
    rows = []
    for arm, load in loaders.items():
        docs = load()
        # the unchunked document, for reference: this is the inflated end
        rows.append({"arm": arm, "size": 0, "fp": round(100 * fp_rate(docs, fakes), 2)})
        print(f"  {arm:3s} unchunked  FP {rows[-1]['fp']:5.2f}%", flush=True)
        for size in args.sizes:
            pieces = chunk(docs, size)
            rows.append(
                {"arm": arm, "size": size, "fp": round(100 * fp_rate(pieces, fakes), 2)}
            )
            print(f"  {arm:3s} @{size:<6} FP {rows[-1]['fp']:5.2f}%", flush=True)
    out = pd.DataFrame(rows).pivot(index="size", columns="arm", values="fp")
    out.to_csv("data/scores/false_positives.csv")
    print("\n=== false positive % on label-swapped rows (size 0 = unchunked) ===")
    print(out.to_string())


if __name__ == "__main__":
    main()
