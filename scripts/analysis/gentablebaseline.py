#!/usr/bin/env python3
"""
Generate_latex_dim_table.py.

Given a CSV with aggregated metrics (one row per embedding method and latent dimension),
generate a LaTeX table showing, for the "noembedding" (baseline) model, how metrics change
across all latent dimensions. Optionally include covariate‐invariance columns for any combination
of age, sex, and/or site.
"""

import argparse

import pandas as pd


def main():
    """Main function for script."""
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table of baseline (noembedding) metrics across latent dims."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input CSV (with baseline metrics across dims).",
    )
    parser.add_argument(
        "--covariates",
        "-c",
        required=True,
        help="Comma‐separated list of covariates to include (none, age, sex, site).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the LaTeX table (e.g. baseline_table.tex).",
    )
    args = parser.parse_args()

    csv_path = args.input
    cov_input = args.covariates.strip().lower()
    out_path = args.output

    # Parse covariates
    if cov_input in {"none", ""}:
        covs = []
    else:
        covs = []
        for c in cov_input.split(","):
            c_clean = c.strip()
            if c_clean not in {"age", "sex", "site"}:
                parser.error(
                    f"Invalid covariate '{c_clean}'. Valid options: age, sex, site, none."
                )
            covs.append(c_clean)
        covs = sorted(set(covs))

    # Load CSV
    df = pd.read_csv(csv_path)

    # Filter to embed == "noembedding"
    if "embed" not in df.columns:
        raise ValueError("Input CSV must have an 'embed' column.")
    df_base = df[df["embed"].str.lower() == "noembedding"].copy()
    if df_base.empty:
        raise ValueError("No rows with embed == 'noembedding' found in the CSV.")

    # Ensure required columns exist
    required = ["dim", "recon_mse_mean", "recon_r2_mean", "global_mean_kl_mean"]
    for col in required:
        if col not in df_base.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")

    # For each requested cov, check *_variant_mean and *_mi_mean exist
    for cov in covs:
        vm = f"{cov}_variant_mean"
        mm = f"{cov}_mi_mean"
        if vm not in df_base.columns or mm not in df_base.columns:
            raise ValueError(
                f"Columns '{vm}' and/or '{mm}' not found for covariate '{cov}'."
            )

    # Sort by dim ascending
    df_base.sort_values("dim", inplace=True)
    df_base.reset_index(drop=True, inplace=True)

    # Build a table of values
    # Columns in order: dim, recon_mse_mean, recon_r2_mean,
    #   for each cov: <cov>_variant_mean, <cov>_mi_mean,
    #   global_mean_kl_mean
    # We'll collect them into a dict of lists to find best values
    dims = df_base["dim"].tolist()
    mse_vals = df_base["recon_mse_mean"].tolist()
    r2_vals = df_base["recon_r2_mean"].tolist()
    kl_vals = df_base["global_mean_kl_mean"].tolist()

    cov_err_vals = {}
    cov_mi_vals = {}
    for cov in covs:
        cov_err_vals[cov] = df_base[f"{cov}_variant_mean"].tolist()
        cov_mi_vals[cov] = df_base[f"{cov}_mi_mean"].tolist()

    # Determine which indices to bold (best per column)
    # MSE: smaller is better → min
    idx_best_mse = mse_vals.index(min(mse_vals))
    # R2: larger is better → max
    idx_best_r2 = r2_vals.index(max(r2_vals))
    # KL: smaller is better → min
    idx_best_kl = kl_vals.index(min(kl_vals))

    # For each cov:
    idx_best_cov_err = {}
    idx_best_cov_mi = {}
    for cov in covs:
        arr_err = cov_err_vals[cov]
        # If cov is "age", that's regression error → smaller is better
        # If cov is "sex" or "site", it's classification accuracy → larger is better
        if cov == "age":
            idx_best_cov_err[cov] = arr_err.index(min(arr_err))
        else:
            idx_best_cov_err[cov] = arr_err.index(max(arr_err))
        # MI: larger is better
        idx_best_cov_mi[cov] = cov_mi_vals[cov].index(max(cov_mi_vals[cov]))

    # Begin building LaTeX lines
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    # Caption and label
    ", ".join([c.capitalize() for c in covs]) if covs else "None"
    caption_txt = (
        "Evaluation metrics of the baseline model across different latent space dimensions ($k$). "
        "Boldface highlights the best (or most desirable) performance per metric."
    )
    lines.append(
        r"\caption[Baseline model performance across latent dimensions]{"
        + caption_txt
        + r"}"
    )
    lines.append(r"\label{tab:baseline_metrics}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Build column format: "c|cc|" + "cc|" per cov + "c"
    col_fmt = "c|cc|" + "".join(["cc|" for _ in covs]) + "c"
    lines.append(r"\begin{tabular}{" + col_fmt + r"}")
    lines.append(r"\toprule")

    # Header row 1
    header_parts = [
        r"\multirow{2}{*}{\textbf{Latent Dim}}",
        r"& \multicolumn{2}{c|}{\textbf{Reconstruction Quality}}",
    ]
    for cov in covs:
        header_parts.append(
            f"& \\multicolumn{{2}}{{c|}}{{\\textbf{{Covariate Invariance ({cov.capitalize()})}}}}"
        )
    header_parts.append(r"& \textbf{Normative Alignment} \\")
    lines.append(" ".join(header_parts))

    # Header row 2
    second_parts = ["& MSE & $R^2$"]
    for cov in covs:
        if cov == "age":
            pred_label = "Age Pred. Error"
        else:
            pred_label = f"{cov.capitalize()} Pred. Accuracy"
        second_parts.append(f"& {pred_label} & MI {cov.capitalize()}")
    second_parts.append(r"& KL Divergence \\")
    lines.append(" ".join(second_parts))

    lines.append(r"\midrule")

    # Data rows
    def fmt(x):
        return f"{x:.2f}" if pd.notna(x) else ""

    for i, k in enumerate(dims):
        # Start with dim
        row_parts = [str(k)]

        # MSE (bold if best)
        mse_str = fmt(mse_vals[i])
        if i == idx_best_mse:
            mse_str = r"\textbf{" + mse_str + "}"
        row_parts += ["&", mse_str]

        # R2 (bold if best)
        r2_str = fmt(r2_vals[i])
        if i == idx_best_r2:
            r2_str = r"\textbf{" + r2_str + "}"
        row_parts += ["&", r2_str]

        # Covariate invariance columns
        for cov in covs:
            # Error or accuracy
            err_str = fmt(cov_err_vals[cov][i])
            if i == idx_best_cov_err[cov]:
                err_str = r"\textbf{" + err_str + "}"
            # MI
            mi_str = fmt(cov_mi_vals[cov][i])
            if i == idx_best_cov_mi[cov]:
                mi_str = r"\textbf{" + mi_str + "}"
            row_parts += ["&", err_str, "&", mi_str]

        # KL Divergence (bold if best)
        kl_str = fmt(kl_vals[i])
        if i == idx_best_kl:
            kl_str = r"\textbf{" + kl_str + "}"
        row_parts += ["&", kl_str, r"\\"]

        lines.append(" ".join(row_parts))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    # Write to output file
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    print(f"LaTeX table across latent dims written to '{out_path}'.")


if __name__ == "__main__":
    main()
