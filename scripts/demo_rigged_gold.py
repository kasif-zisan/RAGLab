"""Why "check true positives and false positives" does NOT catch a rigged gold set.

Builds a gold set the WRONG way - by harvesting tables straight out of arm A's
docling output - and then runs exactly the controls that validate the real
scorer. The rigged set passes every control while handing arm A a score it did
not earn. The controls measure whether the MATCHER is sharp; they cannot see
bias in the NEEDLE, because the gold is the yardstick and a bent yardstick
measures itself as straight.
"""

import json
import random
import re
from pathlib import Path

from raglab.scoring import Haystack, _row_in, gold_rows, span_in

ROW = re.compile(r"^\s*\|.*\|\s*$")
SEP = re.compile(r"^\s*\|[\s:|-]*-[\s:|-]*\|\s*$")


def harvest_tables(stem: str, view: str = "parsed") -> list[str]:
    """Pull whole markdown tables out of a parser's own output, as fake gold."""
    pages = json.load(open(f"data/finqa/{view}/{stem}.json"))["page_markdown"]
    out, block = [], []
    for line in "\n".join(pages).splitlines():
        if ROW.match(line):
            block.append(line)
            continue
        if block:
            body = [ln for ln in block if not SEP.match(ln)]
            if 3 <= len(body) <= 6 and len(body[0].split("|")) >= 4:
                out.append("\n".join(block))
            block = []
    return out


def hay(view: str, stem: str) -> Haystack:
    rec = json.load(open(f"data/finqa/{view}/{stem}.json"))
    return Haystack(" ".join(rec["page_markdown"]))


def main() -> None:
    stems = sorted(p.stem for p in Path("data/finqa/parsed").glob("*.json"))
    random.seed(0)
    sample = random.sample(stems, 60)

    rigged = []
    for stem in sample:
        tables = harvest_tables(stem)
        if tables:
            rigged.append((stem, random.choice(tables)))
    print(f"rigged gold spans harvested from arm A's own output: {len(rigged)}\n")

    # --- the headline score, the thing the rigged set corrupts ---
    for view, label in (("parsed", "A docling"), ("flat", "B flat")):
        cache: dict[str, Haystack] = {}

        def H(stem: str) -> Haystack:
            if stem not in cache:
                cache[stem] = hay(view, stem)
            return cache[stem]

        hit = sum(span_in(sp, H(st)) for st, sp in rigged)
        print(f"  {label:10s} containment on RIGGED gold: {hit}/{len(rigged)} = {hit / len(rigged):.1%}")

    # --- now run Zisan's two checks on the rigged set ---
    print("\nthe two controls, run against the RIGGED gold set:")
    true_rows, fake_rows = [], []
    for stem, span in rigged:
        data = [r for r in gold_rows(span) if len(r) >= 3]
        if len(data) < 2:
            continue
        for i, row in enumerate(data):
            true_rows.append((stem, row))
            other = data[(i + 1) % len(data)]
            if other[0] != row[0]:
                fake_rows.append((stem, [other[0]] + row[1:]))

    for view, label in (("parsed", "A docling"), ("flat", "B flat")):
        cache = {}

        def H(stem: str) -> Haystack:
            if stem not in cache:
                cache[stem] = hay(view, stem)
            return cache[stem]

        t = sum(_row_in(r, H(s), 0) is not None for s, r in true_rows) / len(true_rows)
        f = sum(_row_in(r, H(s), 0) is not None for s, r in fake_rows) / len(fake_rows)
        others = [x for x in sample]
        w = sum(
            span_in(sp, H(random.choice([x for x in others if x != st])))
            for st, sp in rigged
        ) / len(rigged)
        verdict = "PASS" if (f <= 0.02 and w <= 0.02) else "FAIL"
        print(
            f"  {label:10s} true rows kept {t:6.1%}   label-swapped fakes {f:5.1%}"
            f"   wrong-doc {w:5.1%}   -> {verdict}"
        )

    print(
        "\nMeasured outcome: the rigged gold hands arm A 100.0% and arm B 40.0%.\n"
        "The controls land in their NORMAL range (fakes 1.6-3.2%, wrong-doc 0.0%),\n"
        "which reads as 'matcher slightly loose, tighten the window' - not as\n"
        "'this dataset is fabricated'. The bias is invisible to them, because a\n"
        "control compares gold against a haystack, and here the gold IS one arm's\n"
        "haystack. A bent yardstick measures itself as straight.\n"
        "\n"
        "So sensitivity and false-positive checks validate the MATCHER. They say\n"
        "nothing about where the NEEDLE came from, and that is the separate test:\n"
        "could this gold have been derived without looking at any arm's output?"
    )


if __name__ == "__main__":
    main()
