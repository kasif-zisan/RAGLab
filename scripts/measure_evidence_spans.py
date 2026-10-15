"""How big must a chunk be to hold one complete piece of gold evidence?

    uv run python scripts/measure_evidence_spans.py

qa.parquet's source_text column holds the exact text a human marked as
containing the answer, for each of 393 questions. Its length distribution is the
FLOOR for chunk_size: a chunk smaller than a span guarantees the evidence is
split, and neither half alone answers the question.

The ceiling comes from somewhere else entirely - one vector has to represent
everything in the chunk, so a large chunk matches any specific question poorly.
This script only measures the floor.

Note on "fits": chunk_size >= span length is NECESSARY, not sufficient. A span
shorter than the chunk can still straddle a boundary. Overlap mitigates that.
So these numbers are the optimistic case.
"""

import numpy as np
import pandas as pd

QA = "data/finqa/qa.parquet"
CANDIDATES = [256, 400, 512, 800, 1000, 1500, 2000, 3000]


def pct(a, p):
    return np.percentile(a, p)


def main() -> None:
    qa = pd.read_parquet(QA)
    spans = [len(s) for row in qa["source_text"] for s in row]      # every span
    biggest = [max(len(s) for s in row) for row in qa["source_text"]]  # per question
    totals = [sum(len(s) for s in row) for row in qa["source_text"]]

    print(f"{len(qa)} questions, {len(spans)} gold spans\n")

    print("characters per SINGLE span  (the binding constraint - a span must not be split)")
    for p in (50, 75, 90, 95, 99, 100):
        print(f"  p{p:<3} {pct(spans, p):>7,.0f}")
    print(f"  mean {np.mean(spans):>6,.0f}   min {min(spans):>5,}")

    print("\ncharacters per QUESTION, all spans summed (only matters if one chunk must hold all)")
    for p in (50, 90, 99, 100):
        print(f"  p{p:<3} {pct(totals, p):>7,.0f}")

    print("\nlargest span per question - what top_k must cover one of")
    for p in (50, 90, 95, 99):
        print(f"  p{p:<3} {pct(biggest, p):>7,.0f}")

    print("\nfraction of the 393 questions whose LARGEST span could fit in one chunk:")
    print(f"  {'chunk_size':>10}  {'questions covered':>18}  {'~tokens':>8}")
    for c in CANDIDATES:
        ok = sum(b <= c for b in biggest)
        print(f"  {c:>10}  {ok:>7} / {len(biggest)} = {100*ok/len(biggest):5.1f}%  {c//4:>8}")


if __name__ == "__main__":
    main()
