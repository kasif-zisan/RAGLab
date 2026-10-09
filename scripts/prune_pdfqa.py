"""Prune the pdfQA download in place, keeping only the chosen datasets.

Dry run by default - prints what would be removed and what survives:

    uv run python scripts/prune_pdfqa.py

Actually delete (irreversible without re-downloading 25GB):

    uv run python scripts/prune_pdfqa.py --apply

The eval parquet is copied into the project BEFORE anything is deleted. It is
1.7MB, lives outside pdfQA_subset, and holds every question, answer, gold label
and reference context - without it the corpus cannot be evaluated at all.
"""

import argparse
import shutil
from pathlib import Path

SRC = Path("/home/zisan/pdfQA")
SUBSET = SRC / "pdfQA_subset"
REAL = SUBSET / "real-pdfQA"
KEEP = {"FinQA", "FeTaQA"}

# 01.3_Input_Files_CSV is the documents pre-chunked by someone else at ~1.3KB.
# Chunking is milestone 1, so these are actively unwanted, not merely redundant.
FORMAT_DIRS_KEEP = ["01.1_Input_Files_Non_PDF", "01.2_Input_Files_PDF"]
FORMAT_DIRS_DROP = ["01.3_Input_Files_CSV"]

PARQUET = SRC / "real-pdfQA.parquet"
BACKUP = Path("data/real-pdfQA.parquet")


def du(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete")
    args = ap.parse_args()

    if not PARQUET.exists():
        raise SystemExit(f"eval parquet missing at {PARQUET} - refusing to prune")

    # Back up the eval set first, always, even on a dry run.
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(PARQUET, BACKUP)
        print(f"backed up eval set -> {BACKUP} ({BACKUP.stat().st_size/1e6:.1f} MB)\n")
    else:
        print(f"eval set already backed up at {BACKUP}\n")

    targets: list[Path] = []

    # 13GB of git-lfs history duplicating files already in the working tree.
    for g in (SUBSET / ".git", SRC / ".git"):
        if g.exists():
            targets.append(g)

    # Synthetic split - not our corpus.
    for s in (SUBSET / "syn-pdfQA", SRC / "syn-pdfQA", SRC / "syn-pdfQA.parquet"):
        if s.exists():
            targets.append(s)

    # Pre-chunked CSVs.
    for d in FORMAT_DIRS_DROP:
        if (REAL / d).exists():
            targets.append(REAL / d)

    # Unwanted datasets inside the format dirs we keep.
    for d in FORMAT_DIRS_KEEP:
        base = REAL / d
        if not base.is_dir():
            continue
        for ds in sorted(base.iterdir()):
            if ds.is_dir() and ds.name not in KEEP:
                targets.append(ds)

    total = 0
    print(f"{'WOULD REMOVE' if not args.apply else 'REMOVING':<14} size      path")
    for t in targets:
        size = du(t)
        total += size
        print(f"{'':<14} {size/1e9:6.2f} GB  {t.relative_to(SRC)}")
        if args.apply:
            shutil.rmtree(t) if t.is_dir() else t.unlink()

    print(f"\nreclaimed: {total/1e9:.1f} GB")

    if args.apply:
        print(f"remaining: {du(SRC)/1e9:.2f} GB")
        print("\nsurviving corpus:")
        for d in FORMAT_DIRS_KEEP:
            for ds in sorted((REAL / d).iterdir()):
                n = len(list(ds.iterdir()))
                print(f"  {d}/{ds.name:8} {n:4} files  {du(ds)/1e6:7.0f} MB")
    else:
        print("\ndry run - nothing deleted. re-run with --apply to delete.")


if __name__ == "__main__":
    main()
