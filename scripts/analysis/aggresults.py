#!/usr/bin/env python3
"""
Aggresults.py.

Scan through a directory tree of the form:
    experiments/<experiment_folder>/**/validate/output/metrics/metrics.json

Each <experiment_folder> is named like:
    <id>_embed-<embedtype>_dim-<dim>_rep-<rep>_seed-<seed>

    --covariates none
    --covariates age
    --covariates sex,site
    --covariates age,sex,site

to specify which covariates to include.  Valid covariates: age, sex, site.

For each JSON file found, this script collects:
  • recon_mse
  • recon_r2
  • normative_kl.global_mean_kl
  • for each covariate in the user‐specified list:
       – the chosen “variant” metric (normal, nonlinear, neural)
         e.g. invariant_regression_age, invariant_nonlinear_classification_sex, invariant_adversarial_site, etc.
       – the corresponding MI metric: invariant_mi_<cov> → total_mutual_info

Then it groups all repetitions by (embed, dim), computes mean & std across runs, and writes the result to a CSV.

Usage:
    python aggregate_metrics.py \
        --root experiments \
        --variant {normal,nonlinear,neural} \
        --covariates none \
        --output summary.csv
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


def parse_experiment_folder_name(folder_name):
    """
    Given a folder name like:

        0_embed-noembedding_dim-1_rep-0_seed-42

    Extract:
        embed      = "noembedding"
        dim        = 1
        rep        = 0
        seed       = 42
    """
    pattern = (
        r"^[0-9]+_embed-(?P<embed>[^_]+)"
        r"_dim-(?P<dim>\d+)"
        r"_rep-(?P<rep>\d+)"
        r"_seed-(?P<seed>\d+)"
    )
    m = re.match(pattern, folder_name)
    if not m:
        raise ValueError(
            f"Folder name '{folder_name}' does not match expected pattern."
        )
    return {
        "embed": m.group("embed"),
        "dim": int(m.group("dim")),
        "rep": int(m.group("rep")),
        "seed": int(m.group("seed")),
    }


def find_all_metrics_json(root_dir):
    """
    Walk through `root_dir`, find all files named "metrics.json" under.

    any subfolder whose path contains "validate". Yield the full path to each JSON.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        if "validate" not in dirpath:
            continue
        if "metrics.json" in filenames:
            yield os.path.join(dirpath, "metrics.json")


def load_metrics_from_json(json_path):
    """Load a metrics.json and return the parsed dict."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def select_invariant_metric(metrics_dict, cov, variant):
    """If a key or subkey is missing, returns None for that value."""
    if variant == "normal":
        if cov == "age":
            key_variant = "invariant_regression_age"
            subkey = "mse"
        elif cov == "sex":
            key_variant = "invariant_logistic_sex"
            subkey = "accuracy"
        elif cov == "site":
            key_variant = "invariant_logistic_site"
            subkey = "accuracy"
        else:
            return None, None

    elif variant == "nonlinear":
        if cov == "age":
            key_variant = "invariant_nonlinear_regression_age"
            subkey = "mse"
        elif cov == "sex":
            key_variant = "invariant_nonlinear_classification_sex"
            subkey = "accuracy"
        elif cov == "site":
            key_variant = "invariant_nonlinear_classification_site"
            subkey = "accuracy"
        else:
            return None, None

    elif variant == "neural":
        if cov == "age":
            key_variant = "invariant_adversarial_age"
            subkey = "mse"
        elif cov == "sex":
            key_variant = "invariant_adversarial_sex"
            subkey = "accuracy"
        elif cov == "site":
            key_variant = "invariant_adversarial_site"
            subkey = "accuracy"
        else:
            return None, None

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from {{normal, nonlinear, neural}}."
        )

    val_variant = None
    if key_variant in metrics_dict:
        subd = metrics_dict[key_variant]
        if isinstance(subd, dict) and subkey in subd:
            val_variant = subd[subkey]

    mi_key = f"invariant_mi_{cov}"
    val_mi = None
    if mi_key in metrics_dict:
        subd_mi = metrics_dict[mi_key]
        if isinstance(subd_mi, dict) and "total_mutual_info" in subd_mi:
            val_mi = subd_mi["total_mutual_info"]
        elif isinstance(subd_mi, dict) and len(subd_mi) == 1:
            _, vv = next(iter(subd_mi.items()))
            val_mi = vv

    return val_variant, val_mi


def main():
    """Main function for script."""
    parser = argparse.ArgumentParser(
        description="Aggregate JSON‐stored metrics across repeats, grouped by embed & dim."
    )
    parser.add_argument(
        "--root", required=True, help="Path to the top‐level 'experiments' folder"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=["normal", "nonlinear", "neural"],
        help=(
            "Which invariant metric to pick:\n"
            "  normal   → invariant_{regression|logistic}_{cov}\n"
            "  nonlinear→ invariant_nonlinear_{regression|classification}_{cov}\n"
            "  neural   → invariant_adversarial_{cov}"
        ),
    )
    parser.add_argument(
        "--covariates",
        required=True,
        help=(
            "Comma‐separated list of covariates to include: "
            "`none` or any combination of `age`, `sex`, `site`.\n"
            "Examples:\n"
            "  --covariates none\n"
            "  --covariates age\n"
            "  --covariates sex,site\n"
            "  --covariates age,sex,site"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the output CSV (e.g. summary.csv)",
    )
    args = parser.parse_args()

    root_dir = args.root
    variant = args.variant
    output_csv = args.output

    cov_input = args.covariates.strip().lower()
    if cov_input in {"none", ""}:
        covs = []
    else:
        covs = []
        for c in cov_input.split(","):
            c_clean = c.strip()
            if c_clean not in {"age", "sex", "site"}:
                parser.error(
                    f"Invalid covariate '{c_clean}'. Valid options: age, sex, site, or none."
                )
            covs.append(c_clean)
        covs = sorted(set(covs))

    all_records = []

    for json_path in find_all_metrics_json(root_dir):
        folder = Path(json_path).parent
        experiment_folder = None
        for ancestor in (folder, *folder.parents):
            base = ancestor.name
            if re.match(r"^[0-9]+_embed-", base):
                experiment_folder = ancestor
                break

        if experiment_folder is None:
            print(
                f"WARNING: Could not identify experiment folder for {json_path}. Skipping."
            )
            continue

        try:
            parsed = parse_experiment_folder_name(experiment_folder.name)
        except ValueError as e:
            print(f"WARNING: {e} Skipping '{experiment_folder.name}'.")
            continue

        metrics = load_metrics_from_json(json_path)

        record = {
            "embed": parsed["embed"],
            "dim": parsed["dim"],
            "rep": parsed["rep"],
            "seed": parsed["seed"],
            "recon_mse": metrics.get("recon_mse", None),
            "recon_r2": metrics.get("recon_r2", None),
            "global_mean_kl": metrics.get("normative_kl", {}).get(
                "global_mean_kl", None
            ),
        }

        for cov in covs:
            inv_val, mi_val = select_invariant_metric(metrics, cov, variant)
            record[f"{cov}_variant"] = inv_val
            record[f"{cov}_mi"] = mi_val

        all_records.append(record)

    if not all_records:
        print("No metrics.json files were found or parsed. Exiting.")
        return

    df = pd.DataFrame(all_records)

    group_cols = ["embed", "dim"]
    numeric_cols = [c for c in df.columns if c not in group_cols + ["seed", "rep"]]

    agg_dict = {col: ["mean", "std"] for col in numeric_cols}
    df_grouped = df.groupby(group_cols).agg(agg_dict)

    df_grouped.columns = [f"{col}_{stat}" for col, stat in df_grouped.columns]
    df_grouped = df_grouped.reset_index()
    df_grouped.sort_values(by=["embed", "dim"], inplace=True)

    df_grouped.to_csv(output_csv, index=False)
    print(f"Wrote summary CSV with {len(df_grouped)} rows to '{output_csv}'.")


if __name__ == "__main__":
    main()
