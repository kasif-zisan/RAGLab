"""Regression tests for the scorer. Every test here is a bug that actually shipped.

Run:  uv run python -m unittest discover -s tests -v

Deliberately FAST - a handful of documents, not the corpus - so there is no
excuse to skip it. The full-corpus version is scripts/validate_scorer.py and
should be run before trusting any new sweep number.

Written with unittest rather than pytest so it needs no dependency beyond the
standard library. pytest discovers unittest classes unchanged, so this file
works under pytest too if it is ever added.
"""

import json
import unittest
from pathlib import Path

from langchain_core.documents import Document

from raglab.chunking import _SEP, chunk
from raglab.corpus import load_pages
from raglab.scoring import (
    Haystack,
    _prose_in,
    _row_in,
    gold_rows,
    score,
    span_in,
    tokens,
)

DATA = Path("data/finqa")
SAMPLE = ["ORLY_2015", "GIS_2017", "IP_2005", "DRE_2013", "VRTX_2006"]

# The on-disk directory and the arm's view name differ: docling output lives in
# parsed/ but the view key is "docling" (raglab.corpus.VIEWS). Keeping that
# mapping in one place here, because getting it wrong silently produces a
# KeyError deep inside chunk().
VIEW_OF_DIR = {"parsed": "docling", "flat": "flat"}


def hay(view: str, stem: str) -> Haystack:
    rec = json.loads((DATA / view / f"{stem}.json").read_text())
    return Haystack(" ".join(rec["page_markdown"]))


def page_docs(view: str, stems: list[str]) -> list[Document]:
    out = []
    for stem in stems:
        rec = json.loads((DATA / view / f"{stem}.json").read_text())
        for i, page in enumerate(rec["page_markdown"]):
            if page.strip():
                out.append(
                    Document(
                        page_content=page,
                        metadata={
                            "source": stem,
                            "page_index": i,
                            "view": VIEW_OF_DIR[view],
                        },
                    )
                )
    return out


class Normalisation(unittest.TestCase):
    """The gold labels are a different text pipeline's output, not the filing."""

    def test_commas_between_digits_are_removed(self):
        # gold writes 3547, the filing writes 3,547. Fixing only this took prose
        # containment from 0 to 104 of 139.
        self.assertEqual(tokens("3,547"), ["3547"])

    def test_commas_between_words_still_separate_them(self):
        # must NOT glue "x, y" into one token
        self.assertEqual(tokens("alpha, beta"), ["alpha", "beta"])

    def test_html_tags_are_stripped_from_gold(self):
        # the benchmark left raw markup in its labels
        self.assertEqual(
            tokens("Shares<i>(in thousands)</i>", strip_tags=True),
            ["shares", "in", "thousands"],
        )

    def test_tag_regex_cannot_eat_a_less_than_sign(self):
        # bounded to <[^>]{1,20}>, so a stray '<' does not swallow the page
        got = tokens("value < 5 and more text here", strip_tags=True)
        self.assertIn("value", got)
        self.assertIn("more", got)


class GlueMatching(unittest.TestCase):
    """The filing breaks words across lines; the benchmark joined them with no
    space, producing 'Average PricePaid per Share'. Docling renders the space."""

    def setUp(self):
        self.h = Haystack("Average Price Paid per Share 1,234 compensation")

    def test_glued_gold_matches_spaced_filing(self):
        self.assertIsNotNone(self.h.find(["average", "pricepaid"], 0))

    def test_match_must_start_and_end_on_a_token_boundary(self):
        # 'ricepaid' is inside 'price'+'paid' but starts mid-word: must not match
        self.assertIsNone(self.h.find(["ricepaid"], 0))

    def test_de_commaed_number_matches(self):
        self.assertIsNotNone(self.h.find(["1234"], 0))

    def test_prose_is_glue_insensitive_too(self):
        # the scorer used to give this to table cells only, so it PUNISHED
        # docling for correctly rejoining a hyphenated word
        h = Haystack("we expect to pay deferred compensation in fiscal 2018")
        self.assertTrue(_prose_in("we expect to pay deferred com- pensation in fiscal 2018", h))

    def test_prose_still_rejects_a_genuine_mismatch(self):
        h = Haystack("we expect to pay deferred compensation in fiscal 2018")
        self.assertFalse(_prose_in("we expect to pay dividends to holders in fiscal 2019", h))


class FalsifiedRows(unittest.TestCase):
    """A row whose label was swapped for a sibling's: every cell still present,
    the fact now wrong. PyMuPDF dumped the ORLY table column-first, so a loose
    matcher accepts any label with any numbers."""

    HEADER = "|  | December 31, |\n| --- | --- |\n"
    REAL = "| O'Reilly Automotive, Inc. | $100 | $132 | $148 | $213 | $319 | $419 |"
    FAKE = "| S&P 500 Retail Index | $100 | $132 | $148 | $213 | $319 | $419 |"

    def test_real_row_accepted_in_both_parsers(self):
        for view in ("parsed", "flat"):
            with self.subTest(view=view):
                self.assertTrue(span_in(self.HEADER + self.REAL, hay(view, "ORLY_2015")))

    def test_falsified_row_rejected_in_both_parsers(self):
        for view in ("parsed", "flat"):
            with self.subTest(view=view):
                self.assertFalse(span_in(self.HEADER + self.FAKE, hay(view, "ORLY_2015")))

    def test_rows_are_not_found_in_the_wrong_document(self):
        self.assertFalse(span_in(self.HEADER + self.REAL, hay("parsed", "GIS_2017")))

    def test_window_rejects_cells_that_are_far_apart(self):
        # same cells, but with enough filler between them to exceed the window
        h = Haystack("label " + ("filler " * 60) + "100 132 148")
        row = gold_rows("| label | 100 | 132 | 148 |")[0]
        self.assertIsNone(_row_in(row, h, 0))


class ScorerPlumbing(unittest.TestCase):
    """Two bugs where the matcher was correct and the pipeline around it was not."""

    def test_unchunked_documents_do_not_all_score_zero(self):
        # `contained` was derived from chunk_index, which only exists after
        # chunking, so every unchunked run printed 0.0% across the board.
        df = score(page_docs("parsed", SAMPLE))
        self.assertGreater(df.contained.mean(), 0.0)

    def test_prose_prefilter_does_not_block_a_glue_match(self):
        # The prefilter tested exact token membership, so gold 'com'+'pensation'
        # was rejected before the glue-aware matcher ever ran. Symptom: a fix
        # that passed its unit test moved the real metric by 0.000%.
        df = score(page_docs("parsed", ["GIS_2017"]))
        prose = df[(df.source == "GIS_2017") & (~df.is_table)]
        self.assertTrue(prose.contained.any(), "glue match unreachable through score()")

    def test_chunked_scores_are_not_higher_than_unchunked(self):
        # containment is bounded by what the parser produced; chunking can only
        # lose evidence, never create it
        docs = page_docs("parsed", SAMPLE)
        ceiling = score(docs).contained.mean()
        chunked = score(chunk(docs, 1200, 180)).contained.mean()
        self.assertLessEqual(chunked, ceiling + 1e-9)


class ChunkingInvariants(unittest.TestCase):
    def test_separator_row_must_contain_a_dash(self):
        # the regex was [\s:|-]+ which also matched a row of EMPTY cells, so
        # blank rows looked like separators and produced fake "edge cases"
        self.assertTrue(_SEP.match("| --- | --- |"))
        self.assertIsNone(_SEP.match("|  |  |"))

    def test_chunk_refuses_to_mix_arms(self):
        # a score gap is unattributable if two representations share a run
        a = Document(page_content="x", metadata={"source": "s", "view": "docling"})
        b = Document(page_content="y", metadata={"source": "s", "view": "flat"})
        with self.assertRaises(ValueError):
            chunk([a, b], 1200)

    def test_overlap_makes_forward_progress(self):
        # the carry-over guard: never carry the whole buffer, or the packer
        # stalls on one unit and emits forever
        docs = page_docs("parsed", ["ORLY_2015"])
        small = len(chunk(docs, 1200, 0))
        big = len(chunk(docs, 1200, 1000))
        self.assertGreater(small, 0)
        self.assertLess(big, small * 20, "overlap appears to be stalling the packer")

    def test_metadata_survives_chunking(self):
        # without source + page_index on every chunk you cannot cite, and the
        # scorer cannot join to the gold set
        for c in chunk(page_docs("parsed", ["ORLY_2015"]), 1200, 180):
            self.assertIn("source", c.metadata)
            self.assertIn("page_index", c.metadata)
            self.assertIn("chunk_index", c.metadata)


if __name__ == "__main__":
    unittest.main()
