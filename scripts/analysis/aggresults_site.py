#!/usr/bin/env python3
"""
Aggregate_metrics.py.

Scan through a directory tree of the form:
    experiments/<experiment_folder>/**/validate/output/metrics/metrics.json

Each <experiment_folder> is named like:
    <id>_embed-<embedtype>_dim-<dim>_rep-<rep>_seed-<seed>_testsite-<site>

...
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

        2_embed-noembedding_dim-1_rep-0_seed-45_testsite-2

    Extract:
        embed      = "noembedding"
        dim        = 1
        rep        = 0
        seed       = 45
        testsite   = 2
    """
    pattern = (
        r"^[0-9]+_embed-(?P<embed>[^_]+)"
        r"_dim-(?P<dim>\d+)"
        r"_rep-(?P<rep>\d+)"
        r"_seed-(?P<seed>\d+)"
        r"_testsite-(?P<testsite>\d+)$"
    )
    m = re.match(pattern, folder_name)
    if not m:
        raise ValueError(
            f"Folder name '{folder_name}' does not match expected pattern "
            "(including '_testsite-<id>')."
        )

    return {
        "embed": m.group("embed"),
        "dim": int(m.group("dim")),
        "rep": int(m.group("rep")),
        "seed": int(m.group("seed")),
        "testsite": int(m.group("testsite")),
    }


def find_all_metrics_json(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        if "validate" not in dirpath:
            continue
        if "metrics.json" in filenames:
            yield os.path.join(dirpath, "metrics.json")


def load_metrics_from_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def select_invariant_metric(metrics_dict, cov, variant):
    """Returns (variant_value, mi_value) or (-1, -1) if missing or on error."""
    try:
        if variant == "normal":
            if cov == "age":
                key, sub = "invariant_regression_age", "mse"
            elif cov == "sex":
                key, sub = "invariant_logistic_sex", "accuracy"
            elif cov == "site":
                key, sub = "invariant_logistic_site", "accuracy"
            else:
                return -1, -1

        elif variant == "nonlinear":
            if cov == "age":
                key, sub = "invariant_nonlinear_regression_age", "mse"
            elif cov == "sex":
                key, sub = "invariant_nonlinear_classification_sex", "accuracy"
            elif cov == "site":
                key, sub = "invariant_nonlinear_classification_site", "accuracy"
            else:
                return -1, -1

        elif variant == "neural":
            if cov == "age":
                key, sub = "invariant_adversarial_age", "mse"
            elif cov == "sex":
                key, sub = "invariant_adversarial_sex", "accuracy"
            elif cov == "site":
                key, sub = "invariant_adversarial_site", "accuracy"
            else:
                return -1, -1

        else:
            return -1, -1

        # fetch the two numbers, if they exist
        vv = None
        if key in metrics_dict and isinstance(metrics_dict[key], dict):
            vv = metrics_dict[key].get(sub, None)

        mi_key = f"invariant_mi_{cov}"
        mi = None
        if mi_key in metrics_dict and isinstance(metrics_dict[mi_key], dict):
            mi = metrics_dict[mi_key].get(
                "total_mutual_info", next(iter(metrics_dict[mi_key].values()), None)
            )

        # replace any missing ones with -1
        return (vv if vv is not None else -1, mi if mi is not None else -1)

    except Exception:
        return -1, -1


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate JSON‐stored metrics across repeats, grouped by embed, dim, and testsite."
    )
    parser.add_argument(
        "--root", required=True, help="Path to the top‐level 'experiments' folder"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=["normal", "nonlinear", "neural"],
        help="Which invariant metric to pick.",
    )
    parser.add_argument(
        "--covariates",
        required=True,
        help="Comma‐separated list of covariates to include: none or any of age, sex, site.",
    )
    parser.add_argument("--output", required=True, help="Path to write the output CSV")
    args = parser.parse_args()

    cov_input = args.covariates.strip().lower()
    if cov_input in {"none", ""}:
        covs = []
    else:
        covs = sorted(
            {
                c.strip()
                for c in cov_input.split(",")
                if c.strip() in {"age", "sex", "site"}
            }
        )
        if not covs and cov_input not in {"none", ""}:
            parser.error("Invalid covariates. Use 'none' or any of age,sex,site.")

    all_records = []
    for json_path in find_all_metrics_json(args.root):
        folder = Path(json_path).parent
        exp_folder = next(
            (
                anc
                for anc in (folder, *folder.parents)
                if re.match(r"^[0-9]+_embed-", anc.name)
            ),
            None,
        )
        if exp_folder is None:
            print(f"WARNING: no experiment folder found for {json_path}. Skipping.")
            continue

        try:
            parsed = parse_experiment_folder_name(exp_folder.name)
        except ValueError as e:
            print(f"WARNING: {e} Skipping '{exp_folder.name}'.")
            continue

        metrics = load_metrics_from_json(json_path)
        rec = {
            "embed": parsed["embed"],
            "dim": parsed["dim"],
            "rep": parsed["rep"],
            "seed": parsed["seed"],
            "testsite": parsed["testsite"],
            "recon_mse": metrics.get("recon_mse"),
            "recon_r2": metrics.get("recon_r2"),
            "global_mean_kl": metrics.get("normative_kl", {}).get("global_mean_kl"),
        }

        for cov in covs:
            inv_val, mi_val = select_invariant_metric(metrics, cov, args.variant)
            rec[f"{cov}_variant"] = inv_val
            rec[f"{cov}_mi"] = mi_val

        all_records.append(rec)

    if not all_records:
        print("No metrics found. Exiting.")
        return

    df = pd.DataFrame(all_records)

    # 1) Per-testsite aggregation
    group_cols = ["embed", "dim", "testsite"]
    numeric_cols = [c for c in df.columns if c not in group_cols + ["seed", "rep"]]
    agg_dict = {c: ["mean", "std"] for c in numeric_cols}

    df_per_site = df.groupby(group_cols).agg(agg_dict)
    df_per_site.columns = [f"{col}_{stat}" for col, stat in df_per_site.columns]
    df_per_site = df_per_site.reset_index()

    # 2) Overall (across all sites) aggregation under testsite = -1
    df_overall = df.groupby(["embed", "dim"]).agg(agg_dict)
    df_overall.columns = [f"{col}_{stat}" for col, stat in df_overall.columns]
    df_overall = df_overall.reset_index()
    df_overall["testsite"] = -1

    # 3) Combine and sort
    df_combined = pd.concat([df_per_site, df_overall], sort=False)
    df_combined = df_combined[
        ["embed", "dim", "testsite"]
        + sorted(
            c for c in df_combined.columns if c not in {"embed", "dim", "testsite"}
        )
    ]
    df_combined = df_combined.sort_values(by=["embed", "dim", "testsite"]).reset_index(
        drop=True
    )

    df_combined.to_csv(args.output, index=False)
    print(f"Wrote summary CSV with {len(df_combined)} rows to '{args.output}'.")


if __name__ == "__main__":
    main()
