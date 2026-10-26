"""Named pipeline configurations — the single place chunk sizes are decided.

Every script refers to a config by NAME, never by repeating numbers. That is
what makes a result traceable: "recall@5 was 0.81 for B1" means something only
if B1 has exactly one definition, in one file, under version control.

Chosen 2026-10-26 after the free sweeps (21 chunk-size points, 20 overlap
points). See notes/shortlist.md for the full candidate table and notes/
decisions.md for why each number is what it is.

    SELECTED  what we build indexes for now
    SHORTLIST everything the free sweep said was worth paying to test

The two selected configs must be embedded into SEPARATE vector stores. Putting
both in one store would make a retrieval result unattributable - you could not
say whether a hit came from the docling chunks or the flat ones. (A single store
holding both is a different experiment, the "arm C" dual-view idea, and it
answers a different question.)
"""

import json
from dataclasses import dataclass

from langchain_core.documents import Document

from raglab.chunking import chunk
from raglab.corpus import load_filings, load_pages


@dataclass(frozen=True)
class Embedding:
    """Embedding settings — SHARED by every arm, deliberately.

    These are not per-arm fields. The experiment compares parsers and chunking.
    If the arms also differed in embedding settings, a score gap could not be
    attributed to either cause. Everything that is not the thing under test is
    held constant.

    dimensions
        Titan v2 accepts 1024 (default), 512 or 256. AWS reports 512 and 256
        retain 99.0% and 96.8% of 1024's retrieval performance. We use 1024,
        for two reasons. First, dimension does NOT change the bill - Bedrock
        charges per INPUT TOKEN, so a 256-dim vector costs exactly the same to
        generate; the only saving is storage, and storage here is 561 MB across
        both arms, which is nothing. Second, and more important: we are running
        this to MEASURE retrieval quality. The arms differ by ~4 points with a
        +/-3 point uncertainty, so accepting up to 3.2% of dimension-induced
        loss would sit on top of the signal and we could not separate causes.

    normalize
        Defaults to true and is what AWS recommends for RAG. On unit-length
        vectors cosine similarity and inner product rank identically, so search
        can use the cheaper dot product.

    The SAME settings must build the index and the query vectors. A dimension
    mismatch does not degrade gracefully - it either errors or silently returns
    nonsense, which is worse. That is why this lives in one place and why
    `body()` is the only thing that constructs a request.
    """

    model_id: str = "amazon.titan-embed-text-v2:0"
    region: str = "ap-northeast-1"
    dimensions: int = 1024
    normalize: bool = True

    def body(self, text: str) -> str:
        """The exact Bedrock payload. One constructor, so index-time and
        query-time settings cannot drift apart."""
        return json.dumps(
            {
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": self.normalize,
            }
        )

    @property
    def bytes_per_vector(self) -> int:
        return self.dimensions * 4  # float32


EMBED = Embedding()


@dataclass(frozen=True)
class Arm:
    """One complete pipeline setting, from parser through to chunk boundaries."""

    name: str
    view: str  # "docling" (Docling markdown) or "flat" (PyMuPDF text)
    per: str  # "page" or "filing" - what one Document covers before splitting
    chunk_size: int
    overlap: int
    chunks: int  # measured chunk count, so a surprise is visible immediately
    why: str

    def load(self) -> list[Document]:
        if self.per == "filing":
            return load_filings(self.view)
        return load_pages(self.view)

    def build(self) -> list[Document]:
        """Parse-and-split this arm. Deterministic: same inputs, same chunks."""
        return chunk(self.load(), self.chunk_size, self.overlap)

    @property
    def slug(self) -> str:
        """Stable filename stem for this arm's vectors and scores."""
        return f"{self.name}_{self.chunk_size}_ov{self.overlap}"

    def index_mb(self, embed: Embedding = EMBED) -> float:
        """Vector store size for this arm, as float32. Printed before any run
        so an unexpected number is caught before the embedding bill, not after."""
        return self.chunks * embed.bytes_per_vector / 1e6


# What we are building indexes for.
SELECTED: dict[str, Arm] = {
    "A": Arm(
        name="A",
        view="docling",
        per="page",
        chunk_size=2400,
        overlap=360,
        chunks=99_265,
        why=(
            "Docling's plateau. Tables hit 91.0, which IS its parse ceiling, so "
            "this is the best docling can possibly do. Overlap is 15% and makes "
            "no measurable difference for this arm - tables are kept whole by "
            "the splitter, and 38% of its chunks are one entire table, so there "
            "is nothing partial to carry. Kept at 360 rather than 0 only to "
            "protect prose boundaries, worth +0.8 on prose."
        ),
    ),
    "B1": Arm(
        name="B1",
        view="flat",
        per="page",
        chunk_size=3200,
        overlap=889,
        chunks=37_873,
        why=(
            "Leader on containment: tables 94.6 against a 94.9 ceiling, prose "
            "100.0, questions 95.7, on 37,873 chunks - 38% of arm A's count. "
            "Overlap 889 is the p99 of gold evidence length, which is the rule "
            "V >= L (overlap at least as long as the evidence guarantees the "
            "evidence is never cut). Costs only 1.38x the index because large "
            "chunks make overlap cheap: C/(C-V) = 3200/2311."
        ),
    ),
}

# Everything the free sweep said was worth paying to test. Kept because the
# open question is whether SMALL chunks retrieve better once dilution is
# measurable, and that cannot be answered without a small and a large point
# from the same arm.
SHORTLIST: dict[str, Arm] = {
    **SELECTED,
    "A_small": Arm(
        name="A_small", view="docling", per="page", chunk_size=1200, overlap=542,
        chunks=178_051,
        why="Small-chunk docling, for the dilution test. Tables only 72.9.",
    ),
    "B1_small": Arm(
        name="B1_small", view="flat", per="page", chunk_size=1600, overlap=889,
        chunks=94_354,
        why="Small-chunk flat, for the dilution test. Tables 92.8 but 2.25x index.",
    ),
    "B2": Arm(
        name="B2", view="flat", per="filing", chunk_size=2400, overlap=889,
        chunks=51_579,
        why=(
            "Per-filing variant: identical text to B1, split without page "
            "boundaries. Ties B1@3200 on every score (tables 94.6, prose 100.0, "
            "questions 95.7) using 51,579 chunks. Included to test whether page "
            "boundaries matter at all once overlap is set properly."
        ),
    ),
}


def get(name: str) -> Arm:
    if name not in SHORTLIST:
        raise KeyError(f"unknown arm {name!r}; have {sorted(SHORTLIST)}")
    return SHORTLIST[name]
