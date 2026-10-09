"""Inspect a parquet eval set from the terminal.

    uv run python scripts/peek.py                          # schema + dataset counts
    uv run python scripts/peek.py --dataset FinQA          # summary for one dataset
    uv run python scripts/peek.py --dataset FinQA --row 3  # one row in full
"""

import argparse

import pandas as pd

DEFAULT_PATH = "data/real-pdfQA.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--dataset")
    ap.add_argument("--row", type=int, help="print this row in full")
    args = ap.parse_args()

    df = pd.read_parquet(args.path)

    if not args.dataset:
        print(f"{args.path}: {len(df)} rows\n")
        print("schema:")
        for col, dt in df.dtypes.items():
            sample = df[col].iloc[0]
            kind = type(sample).__name__
            print(f"  {col:14} {str(dt):8} -> {kind}")
        print("\nrows per dataset:")
        print(df["dataset"].value_counts().to_string())
        return

    sub = df[df["dataset"] == args.dataset].reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"no rows for {args.dataset!r}; have {sorted(df['dataset'].unique())}")

    if args.row is None:
        print(f"{args.dataset}: {len(sub)} QA pairs over {sub['file_name'].nunique()} gold docs\n")
        print(f"{'#':>3}  {'gold document':38}  question")
        for i, r in sub.head(10).iterrows():
            print(f"{i:>3}  {str(r['file_name'])[:38]:38}  {str(r['question'])[:70]}")
        if len(sub) > 10:
            print(f"     ... {len(sub) - 10} more (use --row N to expand one)")
        return

    r = sub.iloc[args.row]
    print(f"=== {args.dataset} row {args.row} ===\n")
    print(f"gold document : {r['file_name']}")
    print(f"\nquestion      : {r['question']}")
    print(f"\nanswer        : {r['answer']}")
    print(f"\nsource_text   : {len(r['source_text'])} block(s)")
    for j, block in enumerate(r["source_text"]):
        print(f"\n--- block {j} ({len(str(block))} chars) ---")
        print(block)


if __name__ == "__main__":
    main()
