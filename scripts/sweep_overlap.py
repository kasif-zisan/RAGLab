"""Does chunk_overlap buy any evidence containment? Free to measure.

Overlap has exactly one job: repair evidence that a chunk boundary cut in half.
There is a provable rule for how much you need. With chunk size C and overlap V,
chunks start every C-V characters, so evidence of length L is guaranteed to sit
whole inside some chunk when V >= L. Our measured evidence lengths: p50 298,
p90 542, p99 889, max 1163 characters.

So the ladder tested here is 0, our 15% default, p90, and p99.

The cost of overlap is C/(C-V) copies of the corpus embedded, which is why large
chunks make overlap cheap: guaranteeing p99 costs 3.86x at C=1200 but 1.38x at
C=3200. The other cost is not money - heavy overlap fills the top-k with
near-duplicate neighbours, so you retrieve fewer distinct passages.
"""

import argparse
import gc

import pandas as pd

from raglab.chunking import chunk
from raglab.corpus import load_filings, load_pages
from raglab.scoring import score

P90, P99 = 542, 889
POINTS = [("A", 1200), ("A", 2400), ("B1", 1600), ("B1", 3200), ("B2", 2400)]
OUT = "data/scores"


def locatable():
    key = ["qid", "span_index"]
    a = pd.read_parquet(f"{OUT}/A_parse.parquet").set_index(key)["contained"]
    b = pd.read_parquet(f"{OUT}/B1_parse.parquet").set_index(key)["contained"]
    both = a | b
    return both[both].index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=0.75,
                    help="skip overlaps above this fraction of chunk_size")
    args = ap.parse_args()

    loaders = {
        "A": ("docling", lambda: load_pages("docling")),
        "B1": ("flat", lambda: load_pages("flat")),
        "B2": ("flat_pdf", lambda: load_filings("flat")),
    }
    keep = locatable()
    key = ["qid", "span_index"]
    print(f"denominator: {len(keep)} locatable gold spans\n", flush=True)

    rows = []
    docs_cache: dict[str, list] = {}
    for arm, size in POINTS:
        view, load = loaders[arm]
        if arm not in docs_cache:
            docs_cache.clear()
            gc.collect()
            docs_cache[arm] = load()
        docs = docs_cache[arm]
        ladder = sorted({0, int(size * 0.15), P90, P99})
        ladder = [v for v in ladder if v < size * args.cap]
        for ov in ladder:
            pieces = chunk(docs, size, ov)
            n = len(pieces)
            df = score(pieces)
            del pieces
            gc.collect()
            s = df.set_index(key).loc[keep]
            t, p = s[s.is_table], s[~s.is_table]
            cost = size / (size - ov) if ov else 1.0
            rows.append({
                "arm": arm, "size": size, "overlap": ov,
                "chunks": n, "index_cost": round(cost, 2),
                "spans": round(100 * s.contained.mean(), 1),
                "tables": round(100 * t.contained.mean(), 1),
                "prose": round(100 * p.contained.mean(), 1),
                "questions": round(100 * s.groupby("qid")["contained"].all().mean(), 1),
            })
            r = rows[-1]
            print(f"  {arm:3s}@{size:<5} ov={ov:<4} {n:>8,} chunks  x{r['index_cost']:<5}"
                  f" spans {r['spans']:5.1f}  tables {r['tables']:5.1f}"
                  f"  prose {r['prose']:5.1f}  questions {r['questions']:5.1f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/sweep_overlap.csv", index=False)
    print("\n=== overlap sweep, containment % on the locatable denominator ===")
    print(out.to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
