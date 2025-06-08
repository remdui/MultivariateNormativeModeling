#!/usr/bin/env python3
"""Python balanced_sampler.py <N>."""

import random
import sys

import pandas as pd
import pyreadr

##############################################################################
# Hard-coded paths – change if needed
##############################################################################
GENR_RDS = "genr_aparc_vol.rds"
HBN_RDS = "hbn_aparc_vol.rds"
OUTPUT_RDS = "site_dataset.rds"
RANDOM_SEED = 42
##############################################################################


# ---------- helper functions ------------------------------------------------
def read_rds(path):
    """Read RDS."""
    res = pyreadr.read_r(path)
    if not res:
        raise OSError(f"{path} contains no objects")
    return next(iter(res.values()))


def prep(df, site_val, id_col):
    """Prep RDS cols."""
    df = df.copy()
    if site_val is not None:
        df["site"] = site_val
    df["EID"] = df[id_col].astype(str)
    df.drop(
        columns=[c for c in (id_col, "row_id") if c in df.columns],
        inplace=True,
        errors="ignore",
    )
    return df


def split_evenly(total, k):
    """Return a length-k list whose sum == total and differs by ≤1."""
    q, r = divmod(total, k)
    return [q + (1 if i < r else 0) for i in range(k)]


def balanced_sample(df, n, strata_cols, unique_id=None):
    """
    Draw exactly `n` rows, aiming for an even split across the strata.

    If perfect balance isn’t possible, top-up with any remaining rows.
    """
    if unique_id is not None:
        df = df.drop_duplicates(subset=[unique_id])

    # group indices by strata combination
    groups = {k: g.index.to_numpy() for k, g in df.groupby(strata_cols)}

    keys = list(groups.keys())
    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(keys)  # shuffle without NumPy

    targets = split_evenly(n, len(keys))
    chosen = []

    # step 1 – take as even as possible
    for key, target in zip(keys, targets):
        take = min(target, len(groups[key]))
        if take:
            chosen.extend(rnd.sample(list(groups[key]), k=take))

    # step 2 – top-up if still short
    deficit = n - len(chosen)
    if deficit:
        remaining = list(set(df.index) - set(chosen))
        chosen.extend(rnd.sample(remaining, k=deficit))

    # return shuffled dataframe
    return df.loc[chosen].sample(frac=1, random_state=RANDOM_SEED)


# ---------- site-specific wrappers -----------------------------------------


def sample_genr(genr, n):
    """
    • one row per participant (EID) total.

    • roughly equal per wave
    • aim for 50/50 sex *inside each wave*, but guarantee total n
    """
    genr = genr.drop_duplicates(subset=["EID"])

    waves = sorted(genr["wave"].unique())
    targets = split_evenly(n, len(waves))
    rng = random.Random(RANDOM_SEED)

    parts = []
    for wave, tgt in zip(rng.sample(waves, len(waves)), targets):
        wave_df = genr[genr.wave == wave]
        parts.append(balanced_sample(wave_df, tgt, ["sex"]))

    return pd.concat(parts, ignore_index=True)


def sample_hbn(hbn, site_val, n):
    """Balance sex if possible, then top up to reach n."""
    site_df = hbn[hbn.site == site_val]
    return balanced_sample(site_df, n, ["sex"])


def describe_age(df, col="age"):
    """Get age stats."""
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[col].describe()[["count", "mean", "std", "min", "50%", "max"]]


def show_stats(df):
    """Show gathered stats."""
    sep = "=" * 60
    print(sep)
    print(f"Total samples: {len(df)}\n")
    print("Samples per site:\n", df.site.value_counts(), "\n")
    print(
        "Sex per site:\n",
        df.groupby("site").sex.value_counts().unstack(fill_value=0),
        "\n",
    )
    if "age" in df.columns:
        print("Age per site:")
        print(df.groupby("site").apply(lambda x: describe_age(x)).unstack())
        print("\nOverall age:\n", describe_age(df), "\n")
    print("Overall sex:\n", df.sex.value_counts(), "\n")
    print(sep)


def main(n):
    """Main func."""
    random.seed(RANDOM_SEED)

    genr = prep(read_rds(GENR_RDS), site_val=0, id_col="idc")
    hbn = prep(read_rds(HBN_RDS), site_val=None, id_col="EID")  # already has 'site'

    draws = []

    if len(genr) >= n:
        draws.append(sample_genr(genr, n))

    for s in (1, 2, 3):
        if len(hbn[hbn.site == s]) >= n:
            draws.append(sample_hbn(hbn, s, n))

    if not draws:
        raise RuntimeError("No site can supply the requested sample size")

    final = pd.concat(draws, ignore_index=True)

    # Save as .rds
    pyreadr.write_rds(OUTPUT_RDS, final)
    show_stats(final)
    print(f"\nSaved balanced dataset to: {OUTPUT_RDS}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python balanced_sampler.py <n_samples_per_site>")
    try:
        N = int(sys.argv[1])
        if N <= 0:
            raise ValueError
    except ValueError:
        sys.exit("<n_samples_per_site> must be a positive integer")
    main(N)
