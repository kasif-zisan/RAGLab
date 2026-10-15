"""Controls that justify the scorer's two tolerances. Run after ANY change to
raglab.scoring - loosening a matcher is exactly how false positives appear.

1. Mutation control. Take a real gold table row and swap its label for another
   row's label from the same table. Every cell is still present in the document,
   but the fact is now wrong. A matcher that accepts this cannot tell rows
   apart, which is precisely what column-major flat text destroys.
2. Wrong-document control. Test each gold span against a different filing.
3. The ORLY case, by hand. PyMuPDF emitted that performance-graph table
   column-major, so it is the known-hard example.
"""

import json
import random
import sys

import pandas as pd

from raglab.scoring import Haystack, _row_in, gold_rows, span_in


def hay(view: str, stem: str) -> Haystack:
    rec = json.load(open(f"data/finqa/{view}/{stem}.json"))
    return Haystack(" ".join(rec["page_markdown"]))


def main() -> int:
    gold = pd.read_parquet("data/finqa/qa.parquet")
    spans = [(r["file_name"], s) for _, r in gold.iterrows() for s in r["source_text"]]
    stems = sorted({s for s, _ in spans})

    true_rows, fake_rows = [], []
    for _, r in gold.iterrows():
        for span in r["source_text"]:
            data = [x for x in gold_rows(span) if len(x) >= 4]
            if len(data) < 2:
                continue
            for i, row in enumerate(data):
                true_rows.append((r["file_name"], row))
                other = data[(i + 1) % len(data)]
                if other[0] != row[0]:
                    fake_rows.append((r["file_name"], [other[0]] + row[1:]))

    random.seed(0)
    bad = 0
    for view in ("parsed", "flat"):
        cache: dict[str, Haystack] = {}

        def H(stem: str) -> Haystack:
            if stem not in cache:
                cache[stem] = hay(view, stem)
            return cache[stem]

        t = sum(_row_in(r, H(s), 0) is not None for s, r in true_rows) / len(true_rows)
        f = sum(_row_in(r, H(s), 0) is not None for s, r in fake_rows) / len(fake_rows)
        w = sum(
            span_in(sp, H(random.choice([x for x in stems if x != st])))
            for st, sp in spans
        ) / len(spans)
        print(f"{view:7s} true rows kept {t:6.1%}   label-swapped fakes {f:6.1%}   wrong-doc spans {w:6.1%}")
        if f > 0.02 or w > 0.02 or t < 0.85:
            bad += 1

    print()
    header = "|  | December 31, |\n| --- | --- |\n"
    real = "| O'Reilly Automotive, Inc. | $100 | $132 | $148 | $213 | $319 | $419 |"
    fake = "| S&P 500 Retail Index | $100 | $132 | $148 | $213 | $319 | $419 |"
    for view in ("parsed", "flat"):
        h = hay(view, "ORLY_2015")
        ok_real = span_in(header + real, h)
        ok_fake = span_in(header + fake, h)
        print(f"ORLY {view:7s} true row {ok_real!s:5}  FAKE row {ok_fake!s:5}")
        if ok_fake or not ok_real:
            bad += 1

    print("\n" + ("FAIL" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
