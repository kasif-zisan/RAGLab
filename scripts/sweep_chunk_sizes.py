"""The free sweep: gold evidence containment for every arm x chunk size.

Costs nothing - no embeddings, no index, no API calls - so it runs before we
spend money. Its job is to FILTER, not to rank.

Containment rises monotonically with chunk size: a 100,000-character chunk would
contain nearly every gold span. What pushes back is vector dilution (one
embedding representing many topics retrieves worse), and that force is invisible
here - it only appears in recall@k once there is an index. So read this table
looking for the SMALLEST size where the curve flattens, not for the maximum.

Denominator: the gold spans locatable, unchunked, by at least one parser.
The rest are the benchmark's own broken labels and measuring against them
measures FinQA's bugs rather than our pipeline.
"""

import argparse
import gc
import time
from pathlib import Path

import pandas as pd

from raglab.chunking import chunk
from raglab.corpus import load_filings, load_pages
from raglab.scoring import score

SIZES = (800, 1200, 1600, 2000)
OUT = Path("data/scores")


def locatable(key: list[str]) -> pd.Index:
    """Spans findable in the unchunked document by at least one parser."""
    a = pd.read_parquet(OUT / "A_parse.parquet").set_index(key)["contained"]
    b = pd.read_parquet(OUT / "B1_parse.parquet").set_index(key)["contained"]
    both = a | b
    return both[both].index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--arms", nargs="+", default=["A", "B1", "B2"])
    args = ap.parse_args()

    loaders = {
        "A": ("docling", lambda: load_pages("docling")),
        "B1": ("flat", lambda: load_pages("flat")),
        "B2": ("flat_pdf", lambda: load_filings("flat")),
    }
    key = ["qid", "span_index"]
    keep = locatable(key)
    print(f"denominator: {len(keep)} locatable gold spans\n", flush=True)

    rows = []
    for arm in args.arms:
        _, load = loaders[arm]
        docs = load()
        for size in args.sizes:
            name = f"{arm}_{size}"
            t0 = time.time()
            pieces = chunk(docs, size)
            n_chunks = len(pieces)
            df = score(pieces)
            del pieces
            gc.collect()
            df.to_parquet(OUT / f"{name}.parquet")
            s = df.set_index(key).loc[keep]
            t, p = s[s.is_table], s[~s.is_table]
            rows.append(
                {
                    "arm": arm,
                    "size": size,
                    "chunks": n_chunks,
                    "spans": round(100 * s.contained.mean(), 1),
                    "tables": round(100 * t.contained.mean(), 1),
                    "prose": round(100 * p.contained.mean(), 1),
                    "questions": round(
                        100 * s.groupby("qid")["contained"].all().mean(), 1
                    ),
                    "secs": round(time.time() - t0),
                }
            )
            print(f"  {name:9s} {n_chunks:>8,} chunks  "
                  f"spans {rows[-1]['spans']:5.1f}  tables {rows[-1]['tables']:5.1f}  "
                  f"prose {rows[-1]['prose']:5.1f}  questions {rows[-1]['questions']:5.1f}  "
                  f"({rows[-1]['secs']}s)", flush=True)
        del docs
        gc.collect()

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "sweep.csv", index=False)
    print("\n=== containment %, on the locatable denominator ===")
    print(out.to_string(index=False))
    print(f"\nwrote {OUT / 'sweep.csv'}")


if __name__ == "__main__":
    main()
