"""Split preprocessed dataset on site."""

import os
import sys

import pandas as pd

# number of extra samples to pull from each non-held-out site
SAMPLES_PER_SITE = 100
# for reproducibility
RANDOM_SEED_START = 42


def main(input_csv: str):
    """Main func."""
    df = pd.read_csv(input_csv)

    # strip trailing "_train"/"_test" from the basename
    base = os.path.splitext(os.path.basename(input_csv))[0]
    if base.endswith("_train"):
        root = base[:-6]
    elif base.endswith("_test"):
        root = base[:-5]
    else:
        root = base

    # find your one-hot columns, e.g. "site_0.0"
    site_cols = [c for c in df.columns if c.startswith("site_")]
    if not site_cols:
        print("ERROR: no 'site_' columns found in", input_csv)
        sys.exit(1)

    for col in site_cols:
        # parse out the integer site ID from "0.0"
        suffix = col.split("_", 1)[1]
        try:
            site_id = int(float(suffix))
        except ValueError:
            print(f"Skipping column {col!r}: can't parse site ID")
            continue

        # 1) primary test = all rows of this site
        primary_test = df[df[col] == 1]
        # 2) initial train pool = rows *not* of this site
        train_pool = df[df[col] == 0]

        # 3) from each of the *other* one-hot columns,
        #    sample up to SAMPLES_PER_SITE rows
        extra_pieces = []
        seed = RANDOM_SEED_START
        others = [c for c in site_cols if c != col]
        for oc in others:
            subset = train_pool[train_pool[oc] == 1]
            n_avail = len(subset)
            n_take = min(SAMPLES_PER_SITE, n_avail)
            if n_avail < SAMPLES_PER_SITE:
                print(f"Warning: only {n_avail} rows in {oc}, sampling all of them")
            sampled = subset.sample(n=n_take, random_state=seed)
            extra_pieces.append(sampled)
            # bump seed so each site gets a different draw
            seed += 1

        extra_df = pd.concat(extra_pieces, axis=0) if extra_pieces else pd.DataFrame()

        # final test = primary_test + these extra samples
        test_df = pd.concat([primary_test, extra_df], axis=0)
        # final train = train_pool minus the extra samples
        train_df = train_pool.drop(extra_df.index)

        # write out
        train_fname = f"{root}_site_{site_id}_train.csv"
        test_fname = f"{root}_site_{site_id}_test.csv"

        train_df.to_csv(train_fname, index=False)
        test_df.to_csv(test_fname, index=False)

        print(f"→ {train_fname}: {len(train_df)} rows")
        print(f"→ {test_fname} : {len(test_df)} rows\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_by_site.py <input_csv>")
        sys.exit(1)
    main(sys.argv[1])
