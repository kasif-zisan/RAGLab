"""Stage selected pdfQA-Benchmark datasets into data/.

    uv run python scripts/stage_corpus.py                      # FinQA gold-only + FeTaQA full
    uv run python scripts/stage_corpus.py --datasets FinQA --all-docs
    uv run python scripts/stage_corpus.py --datasets FeTaQA --no-pdf

"Gold" is per question: with 393 questions over 196 gold docs, a gold-only FinQA
corpus still gives each question 195 distractors - and they are hard negatives
(same domain, structure and vocabulary). So gold-only is a real retrieval task,
not a degenerate one, and it costs ~49k chunks instead of ~256k.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

SRC = Path("/home/zisan/pdfQA")
REAL = SRC / "pdfQA_subset" / "real-pdfQA"
TXT_SRC = REAL / "01.1_Input_Files_Non_PDF"
PDF_SRC = REAL / "01.2_Input_Files_PDF"
QA_PATH = Path("data/real-pdfQA.parquet")
DST = Path("data")

# FinQA gold-only keeps the index affordable; FeTaQA is small enough to take whole.
DEFAULTS = {"FinQA": dict(gold_only=True, pdf=True),
            "FeTaQA": dict(gold_only=False, pdf=True)}


def stage(name: str, qa_all: pd.DataFrame, gold_only: bool, want_pdf: bool) -> dict:
    qa = qa_all[qa_all["dataset"] == name].reset_index(drop=True)
    if qa.empty:
        raise SystemExit(f"no QA rows for dataset {name!r}")
    qa["source_text"] = qa["source_text"].apply(list)
    gold = set(qa["file_name"])

    out = DST / name.lower()
    out.mkdir(parents=True, exist_ok=True)
    qa.to_parquet(out / "qa.parquet", index=False)

    def wanted(stem: str) -> bool:
        return (stem in gold) if gold_only else True

    txt_dst = out / "text_pymupdf"
    txt_dst.mkdir(exist_ok=True)
    kept_txt = []
    for src in sorted((TXT_SRC / name).glob("*.txt")):
        stem = src.name.replace("__pymupdf.txt", "")
        if wanted(stem):
            shutil.copy2(src, txt_dst / src.name)
            kept_txt.append(stem)

    kept_pdf = []
    if want_pdf:
        pdf_dst = out / "pdf"
        pdf_dst.mkdir(exist_ok=True)
        for src in sorted((PDF_SRC / name).glob("*.pdf")):
            if wanted(src.stem):
                shutil.copy2(src, pdf_dst / src.name)
                kept_pdf.append(src.stem)

    # The corpus manifest. Every experiment must load the document list from here,
    # never by globbing the directory - a corpus that changes between runs makes
    # two experiments incomparable.
    manifest = dict(dataset=name, gold_only=gold_only, qa_pairs=len(qa),
                    gold_docs=sorted(gold), corpus_docs=sorted(kept_txt))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    missing = sorted(gold - set(kept_txt)) if gold_only else []
    return dict(dataset=name, qa=len(qa), gold=len(gold), docs=len(kept_txt),
                distractors=len(kept_txt) - len(gold), pdfs=len(kept_pdf),
                MB=round(mb), missing_gold=len(missing))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULTS))
    ap.add_argument("--all-docs", action="store_true", help="include distractors")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    qa_all = pd.read_parquet(QA_PATH)
    if unknown := set(args.datasets) - set(qa_all["dataset"]):
        raise SystemExit(f"unknown dataset(s): {sorted(unknown)}")

    report = []
    for n in args.datasets:
        d = DEFAULTS.get(n, dict(gold_only=False, pdf=True))
        report.append(stage(n, qa_all,
                            gold_only=d["gold_only"] and not args.all_docs,
                            want_pdf=d["pdf"] and not args.no_pdf))

    print(pd.DataFrame(report).to_string(index=False))
    if any(r["missing_gold"] for r in report):
        raise SystemExit("ERROR: gold documents missing from corpus - eval would be invalid")
    print(f"\ntotal staged: {sum(r['MB'] for r in report)} MB")


if __name__ == "__main__":
    main()
