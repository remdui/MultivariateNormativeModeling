"""Split train data in different train test subsets based on site."""

#!/usr/bin/env python3

import os
import sys

import pandas as pd


def main(input_csv: str):
    """Main func."""
    # Load the full CSV
    df = pd.read_csv(input_csv)

    # Figure out the root name (drop a trailing "_train" or "_test" if present)
    base = os.path.splitext(os.path.basename(input_csv))[0]
    if base.endswith("_train"):
        root = base[: -len("_train")]
    elif base.endswith("_test"):
        root = base[: -len("_test")]
    else:
        root = base

    # Detect one-hot site columns like "site_0.0", "site_1.0", …
    site_cols = [c for c in df.columns if c.startswith("site_")]
    if not site_cols:
        print("ERROR: no 'site_' columns found in", input_csv)
        sys.exit(1)

    for col in site_cols:
        # parse the numeric part (e.g. "0.0" → 0)
        suffix = col.split("_", 1)[1]
        try:
            site_id = int(float(suffix))
        except ValueError:
            print(f"Skipping column {col!r}: cannot parse site id")
            continue

        # split off test rows (where this col==1) and train rows (col==0)
        test_df = df[df[col] == 1]
        train_df = df[df[col] == 0]

        # build and write files
        test_fname = f"{root}_site_{site_id}_test.csv"
        train_fname = f"{root}_site_{site_id}_train.csv"

        test_df.to_csv(test_fname, index=False)
        train_df.to_csv(train_fname, index=False)

        print(f"→ {train_fname}: {len(train_df)} rows")
        print(f"→ {test_fname} : {len(test_df)} rows")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_by_site.py <input_csv>")
        sys.exit(1)
    main(sys.argv[1])
