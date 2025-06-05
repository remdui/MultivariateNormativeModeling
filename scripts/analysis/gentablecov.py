#!/usr/bin/env python3
"""
Generate_latex_table.py.

Given a CSV with aggregated metrics (one row per embedding method at a fixed latent dimension),
generate a LaTeX table that can include multiple covariates side‐by‐side.
"""

import argparse
from collections import OrderedDict

import pandas as pd


def main():
    """Main function for script."""
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table of covariate modeling results from an aggregated CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input CSV. Must have columns: embed, dim, recon_mse_mean, recon_r2_mean, "
        "<cov>_variant_mean, <cov>_mi_mean (for each requested cov), global_mean_kl_mean.",
    )
    parser.add_argument(
        "--covariates",
        "-c",
        required=True,
        help="Comma‐separated list of covariates to include (e.g. 'none', 'age', 'sex,site', 'age,sex,site').",
    )
    parser.add_argument(
        "--dim", "-d", type=int, required=True, help="Latent dimensionality k (e.g. 8)."
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the LaTeX table (e.g. table_multi_cov.tex).",
    )
    args = parser.parse_args()

    csv_path = args.input
    cov_input = args.covariates.strip().lower()
    dim = args.dim
    out_path = args.output

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
        covs = sorted(set(covs))  # remove duplicates, sort alphabetically

    df = pd.read_csv(csv_path)

    df_dim = df[df["dim"] == dim].copy()
    if df_dim.empty:
        raise ValueError(f"No rows found with dim == {dim} in '{csv_path}'.")

    base_cols = [
        "embed",
        "dim",
        "recon_mse_mean",
        "recon_r2_mean",
        "global_mean_kl_mean",
    ]
    for col in base_cols:
        if col not in df_dim.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")

    for cov in covs:
        vm = f"{cov}_variant_mean"
        mm = f"{cov}_mi_mean"
        if vm not in df_dim.columns or mm not in df_dim.columns:
            raise ValueError(
                f"Columns '{vm}' and/or '{mm}' not found for covariate '{cov}'."
            )

    row_dict = {}
    for _, row in df_dim.iterrows():
        key = row["embed"].strip()
        entry = {
            "mse": row["recon_mse_mean"],
            "r2": row["recon_r2_mean"],
            "kl": row["global_mean_kl_mean"],
        }
        for cov in covs:
            entry[f"{cov}_err"] = row[f"{cov}_variant_mean"]
            entry[f"{cov}_mi"] = row[f"{cov}_mi_mean"]
        row_dict[key] = entry

    # Hard‐coded order and label mapping from embed -> LaTeX row label
    methods = OrderedDict(
        [
            ("noembedding", "Baseline"),
            ("decoderembedding", "Decoder-only"),
            ("encoderembedding", "Encoder-only"),
            ("encoderdecoderembedding", "Encoder-Decoder (cVAE)"),
            ("inputfeatureembedding", "Covariate reconstruction"),
            ("conditionalembedding", "Conditional loss term"),
            ("adversarialembedding", "Adversarial loss term"),
            ("conditionaladversarialembedding", "Conditional Adversarial loss term"),
            ("fairembedding", "FairVAE (MMD)"),
            ("hsicembedding", "HSIC loss term"),
            ("disentangleembedding", "Disentangled subspace"),
        ]
    )

    lines = []
    cov_titles = [c.capitalize() for c in covs]  # e.g. ["Age", "Sex"]

    cov_list_title = ", ".join(cov_titles) if cov_titles else "None"
    caption_txt = (
        f"Evaluation of different covariate modeling strategies when modeling "
        f"{cov_list_title} at latent dimensionality $k={dim}$."
    )
    label_txt = f"tab:covariate_modeling_{cov_input.replace(',', '_')}"

    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption[Covariate modeling strategies for {cov_list_title}]{{{caption_txt}}}"
    )
    lines.append(f"\\label{{{label_txt}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    col_fmt = "l|cc|" + "".join(["cc|" for _ in covs]) + "c"
    lines.append(r"\begin{tabular}{" + col_fmt + r"}")
    lines.append(r"\toprule")

    header_parts = [
        r"\multirow{2}{*}{\textbf{Covariate Modeling Method}}",
        r"& \multicolumn{2}{c|}{\textbf{Reconstruction Quality}}",
    ]
    for ct in cov_titles:
        header_parts.append(
            f"& \\multicolumn{{2}}{{c|}}{{\\textbf{{Covariate Invariance ({ct})}}}}"
        )
    header_parts.append(r"& \textbf{Normative Alignment} \\")
    lines.append(" ".join(header_parts))

    second_parts = ["& MSE & $R^2$"]
    for ct in cov_titles:
        second_parts.append(f"& {ct} Pred. Error & MI {ct}")
    second_parts.append(r"& KL Divergence \\")
    lines.append(" ".join(second_parts))

    lines.append(r"\midrule")

    def fmt(x):
        return f"{x:.2f}" if pd.notna(x) else ""

    for embed_key, label in methods.items():
        if embed_key in row_dict:
            vals = row_dict[embed_key]
            mse_val = fmt(vals["mse"])
            r2_val = fmt(vals["r2"])
            kl_val = fmt(vals["kl"])
            row_parts = [f"{label:<30}", "&", mse_val, "&", r2_val]
            for cov in covs:
                err_val = fmt(vals[f"{cov}_err"])
                mi_val = fmt(vals[f"{cov}_mi"])
                row_parts += ["&", err_val, "&", mi_val]
            row_parts += ["&", kl_val, r"\\"]
            lines.append(" ".join(row_parts))
        else:
            blanks = [""] * (2 + 2 * len(covs) + 1)  # mse,r2 + (err,mi)*nc + kl
            row_parts = [f"{label:<30}", "&", blanks[0], "&", blanks[1]]
            idx = 2
            for _ in covs:
                row_parts += ["&", blanks[idx], "&", blanks[idx + 1]]
                idx += 2
            row_parts += ["&", blanks[-1], r"\\"]
            lines.append(" ".join(row_parts))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    print(f"LaTeX table with covariates [{cov_list_title}] written to '{out_path}'.")


if __name__ == "__main__":
    main()
