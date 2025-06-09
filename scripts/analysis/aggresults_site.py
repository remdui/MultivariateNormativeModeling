#!/usr/bin/env python3
"""
Generate_latex_table.py.

Given a CSV with aggregated metrics (one row per embedding method, dim, testsite),
produce a LaTeX table showing for each embedding method four rows: the results for
site=0, site=1, site=2, and the average across all sites (site=-1) in bold.
Rows are indented so that the test site label is one column deeper than the method.
Each table is generated for a specific latent dim passed on the command line.
"""

import argparse
from collections import OrderedDict

import pandas as pd


def main():
    """Main func."""
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table of per-site and overall metrics for each embedding."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input CSV. Must have columns: embed, dim, testsite, recon_mse_mean, recon_r2_mean, global_mean_kl_mean, <cov>_variant_mean, <cov>_mi_mean.",
    )
    parser.add_argument(
        "--covariates",
        "-c",
        required=True,
        help="Comma-separated list of covariates to include (e.g. 'none', 'age', 'sex,site', 'age,sex,site').",
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

    # Parse covariates
    cov_input = args.covariates.strip().lower()
    if cov_input in {"none", ""}:
        covs = []
    else:
        covs = []
        for c in cov_input.split(","):
            c_clean = c.strip()
            if c_clean not in {"age", "sex", "site"}:
                parser.error(
                    f"Invalid covariate '{c_clean}'. Valid: age, sex, site, or none."
                )
            covs.append(c_clean)
        covs = sorted(set(covs))

    # Read CSV and filter by dim
    df = pd.read_csv(args.input)
    df = df[df["dim"] == args.dim].copy()
    if df.empty:
        raise ValueError(f"No rows found with dim == {args.dim} in '{args.input}'")

    # Ensure required columns exist
    base_cols = [
        "embed",
        "testsite",
        "recon_mse_mean",
        "recon_r2_mean",
        "global_mean_kl_mean",
    ]
    for col in base_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")
    for cov in covs:
        vm = f"{cov}_variant_mean"
        mm = f"{cov}_mi_mean"
        if vm not in df.columns or mm not in df.columns:
            raise ValueError(
                f"Columns '{vm}' and/or '{mm}' not found for covariate '{cov}'."
            )

    # Row data structure: { embed: { testsite: { metrics... } } }
    data = {}
    for _, row in df.iterrows():
        embed = row["embed"].strip()
        site = int(row["testsite"])
        if embed not in data:
            data[embed] = {}
        entry = {
            "mse": row["recon_mse_mean"],
            "r2": row["recon_r2_mean"],
            "kl": row["global_mean_kl_mean"],
        }
        for cov in covs:
            entry[f"{cov}_err"] = row[f"{cov}_variant_mean"]
            entry[f"{cov}_mi"] = row[f"{cov}_mi_mean"]
        data[embed][site] = entry

    # Fixed order of methods and labels
    methods = OrderedDict(
        [
            ("noembedding", "Baseline"),
            ("encoderdecoderembedding", "Encoder-Decoder (cVAE)"),
            ("fairembedding", "FairVAE (MMD)"),
            ("harmonized", "Combat Harmonized"),
        ]
    )

    # Friendly labels for site IDs
    site_names = {0: "GenR", 1: "RUBIC", 2: "CBIC"}

    # Build LaTeX
    lines = []
    cov_titles = [c.capitalize() for c in covs]
    # Caption & label
    cov_list_title = ", ".join(cov_titles) if cov_titles else "None"
    dim = args.dim
    caption = (
        f"Evaluation of different covariate modeling strategies for {cov_list_title} "
        f"at latent dimension $k={dim}$."
    )
    label = f"tab:covariate_modeling_{cov_input.replace(',', '_')}"

    # Table begin
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption[{cov_list_title}]{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column formatting: method, site, recon(2), each cov(2), kl
    col_fmt = "l c|cc|" + "".join(["cc|" for _ in covs]) + "c"
    lines.append(r"\begin{tabular}{" + col_fmt + r"}")
    lines.append(r"\toprule")

    # Header rows
    hdr1 = [
        r"\multirow{2}{*}{\textbf{Method}}",
        r"\multirow{2}{*}{\textbf{Test Site}}",
        r"\multicolumn{2}{c|}{\textbf{Reconstruction Quality}}",
    ]
    for ct in cov_titles:
        hdr1.append(rf"\multicolumn{{2}}{{c|}}{{\textbf{{Invariance ({ct})}}}}")
    hdr1.append(r"\multirow{2}{*}{\textbf{Normative Alignment}}")
    lines.append(" & ".join(hdr1) + r" \\")

    hdr2 = ["", "", "MSE", r"$R^2$"]
    for ct in cov_titles:
        hdr2 += [f"{ct} Err", f"MI {ct}"]
    hdr2.append("")
    lines.append(" & ".join(hdr2) + r" \\")

    lines.append(r"\midrule")

    def fmt(x):
        return f"{x:.2f}" if pd.notna(x) else ""

    # For each method, output 4 subrows
    for embed_key, label_txt in methods.items():
        if embed_key not in data:
            continue
        subsites = [0, 1, 2, -1]
        for i, site in enumerate(subsites):
            # prepare entry (site data or average)
            if site == -1:
                vals = [data[embed_key].get(s, {}) for s in (0, 1, 2)]
                entry = {
                    "mse": pd.Series([v.get("mse", float("nan")) for v in vals]).mean(),
                    "r2": pd.Series([v.get("r2", float("nan")) for v in vals]).mean(),
                    "kl": pd.Series([v.get("kl", float("nan")) for v in vals]).mean(),
                }
                for cov in covs:
                    entry[f"{cov}_err"] = pd.Series(
                        [v.get(f"{cov}_err", float("nan")) for v in vals]
                    ).mean()
                    entry[f"{cov}_mi"] = pd.Series(
                        [v.get(f"{cov}_mi", float("nan")) for v in vals]
                    ).mean()
                sublabel = r"\textbf{Average}"
            else:
                entry = data[embed_key].get(site, {})
                sublabel = site_names.get(site, f"Site {site}")

            # build row
            if i == 0:
                row = rf"\multirow{{4}}{{*}}{{{label_txt}}} & {sublabel} & {fmt(entry.get('mse'))} & {fmt(entry.get('r2'))}"
            else:
                row = (
                    f" & {sublabel} & {fmt(entry.get('mse'))} & {fmt(entry.get('r2'))}"
                )

            # covariate columns
            for cov in covs:
                err = fmt(entry.get(f"{cov}_err"))
                mi = fmt(entry.get(f"{cov}_mi"))
                row += f" & {err} & {mi}"

            # KL divergence + end of row
            row += f" & {fmt(entry.get('kl'))} \\\\"
            lines.append(row)

        # rule between methods
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}\end{table}")

    # Write to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table written to '{args.output}'")


if __name__ == "__main__":
    main()
