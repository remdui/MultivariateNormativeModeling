"""Merge and process GENR MRI data from FreeSurfer output files."""

import logging
import os
from itertools import combinations

import numpy as np
import pandas as pd
import pyreadr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Constants
OUTPUT_DIR = "../../../data"
WAVE_MAPPING = {"f05": 0, "f09": 1, "f13": 2}
APARC_FILES = [
    "f05_freesurfer_v6_24june2021_aparc_stats_pull18Aug2021.rds",
    "f09_freesurfer_v6_09dec2016_aparc_stats_pull06june2017.rds",
    "f13_freesurfer_v6_14oct2020_aparc_stats_pull23Nov2020.rds",
]
ASEG_FILES = [
    "f05_freesurfer_v6_24june2021_aseg_stats_pull18Aug2021_v1.rds",
    "f09_freesurfer_v6_09dec2016_aseg_stats_pull06june2017_v1.rds",
    "f13_freesurfer_v6_14oct2020_aseg_stats_pull23Nov2020_v1.rds",
]
CORE_FILE = "genr_mri_core_data_20231204.rds"

# Features to exclude
APARC_EXCLUDE_FEATURES = ["lh_MeanThickness", "rh_MeanThickness"]
ASEG_EXCLUDE_FEATURES = []

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_core_data(file_path):
    """Load the core data file and process the covariates."""
    core_data = pyreadr.read_r(file_path)[None]
    covariates = core_data[
        ["idc", "age_child_mri_f05", "age_child_mri_f09", "age_child_mri_f13"]
    ]
    covariates = pd.melt(
        covariates, id_vars=["idc"], var_name="wave", value_name="age"
    ).dropna()
    covariates["wave"] = covariates["wave"].str[-3:].map(WAVE_MAPPING)
    covariates["age"] = covariates["age"].astype(np.float64)
    logging.info(
        f"Core data loaded with {covariates.shape[0]} rows and columns: {list(covariates.columns)}"
    )
    return covariates


def load_and_process_files(file_list, wave_mapping, is_aseg=False):
    """Load and process the FreeSurfer output files."""
    processed_data = []
    exclude_features = ASEG_EXCLUDE_FEATURES if is_aseg else APARC_EXCLUDE_FEATURES
    for file_path in file_list:
        wave_label = os.path.basename(file_path)[:3]
        data = pyreadr.read_r(file_path)[None]
        data.rename(
            columns=lambda col: col.rstrip("_f05").rstrip("_f09").rstrip("_f13"),
            inplace=True,
        )
        if is_aseg:
            data.rename(
                columns=lambda col: col[:-4] if col.endswith("_vol") else col,
                inplace=True,
            )
        data = data.drop(
            columns=[col for col in exclude_features if col in data.columns],
            errors="ignore",
        )
        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    {"wave": [wave_mapping[wave_label]] * len(data)}, index=data.index
                ),
            ],
            axis=1,
        )
        data = data.astype(np.float64, errors="ignore")
        processed_data.append(data)
    combined_data = pd.concat(processed_data, ignore_index=True).dropna()
    return combined_data


def combine_hemisphere_columns(df):
    """Combine the left and right hemisphere columns."""
    hemisphere_prefixes = ["lh_", "rh_"]
    combined_columns = {}
    for prefix in hemisphere_prefixes:
        for col in [c[len(prefix) :] for c in df.columns if c.startswith(prefix)]:
            combined_columns[col] = (
                df[[f"lh_{col}", f"rh_{col}"]].mean(axis=1).astype(np.float64)
            )
    drop_cols = [f"lh_{col}" for col in combined_columns] + [
        f"rh_{col}" for col in combined_columns
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    combined_df = pd.DataFrame(combined_columns, index=df.index)
    return pd.concat([df, combined_df], axis=1)


def save_dataframe(df, file_path, description):
    """Save the dataframe to an RDS file."""
    df = df.reset_index().rename(columns={"index": "row_id"})
    pyreadr.write_rds(file_path, df)
    logging.info(
        f"{description} saved to {file_path} with {df.shape[0]} rows and columns: {list(df.columns)}\n\n"
    )


def process_and_save_subsets(df, output_dir, prefix, covariates):
    """Process and save subsets of the data."""
    subsets = {
        "surfarea": [col for col in df.columns if "surfarea" in col.lower()],
        "vol": [col for col in df.columns if "vol" in col.lower()],
        "thickavg": [col for col in df.columns if "thickavg" in col.lower()],
    }

    # Process individual subsets
    for subset_name, subset_cols in subsets.items():
        subset_data = df[["idc", "wave"] + subset_cols].dropna()
        combined_data = combine_hemisphere_columns(subset_data)
        merged_data = covariates.merge(subset_data, on=["idc", "wave"], how="inner")
        combined_merged_data = covariates.merge(
            combined_data, on=["idc", "wave"], how="inner"
        )

        save_dataframe(
            merged_data,
            os.path.join(output_dir, f"{prefix}_{subset_name}_lh_rh.rds"),
            f"{prefix} {subset_name} (lh/rh)",
        )
        save_dataframe(
            combined_merged_data,
            os.path.join(output_dir, f"{prefix}_{subset_name}.rds"),
            f"{prefix} {subset_name} (combined)",
        )

    # Process pair combinations of subsets
    subset_names = list(subsets.keys())
    for pair in combinations(subset_names, 2):
        pair_name = "_and_".join(pair)
        pair_columns = list(set(subsets[pair[0]] + subsets[pair[1]]))
        pair_data = df[["idc", "wave"] + pair_columns].dropna()
        combined_pair_data = combine_hemisphere_columns(pair_data)
        merged_pair_data = covariates.merge(pair_data, on=["idc", "wave"], how="inner")
        combined_merged_pair_data = covariates.merge(
            combined_pair_data, on=["idc", "wave"], how="inner"
        )

        save_dataframe(
            merged_pair_data,
            os.path.join(output_dir, f"{prefix}_{pair_name}_lh_rh.rds"),
            f"{prefix} {pair_name} (lh/rh)",
        )
        save_dataframe(
            combined_merged_pair_data,
            os.path.join(output_dir, f"{prefix}_{pair_name}.rds"),
            f"{prefix} {pair_name} (combined)",
        )


def process_and_save_with_combined(
    aparc_data, aseg_data, full_data, covariates, output_dir
):
    """Process and save the data with combined hemisphere columns."""
    aparc_combined = combine_hemisphere_columns(aparc_data)
    # aseg_combined = combine_hemisphere_columns(aseg_data)
    combined_full_data = combine_hemisphere_columns(full_data)

    aparc_with_covariates = covariates.merge(
        aparc_data, on=["idc", "wave"], how="inner"
    )
    aseg_with_covariates = covariates.merge(aseg_data, on=["idc", "wave"], how="inner")
    full_with_covariates = covariates.merge(full_data, on=["idc", "wave"], how="inner")

    aparc_combined_with_covariates = covariates.merge(
        aparc_combined, on=["idc", "wave"], how="inner"
    )
    # aseg_combined_with_covariates = covariates.merge(aseg_combined, on=["idc", "wave"], how="inner")
    combined_full_with_covariates = covariates.merge(
        combined_full_data, on=["idc", "wave"], how="inner"
    )

    save_dataframe(
        aparc_with_covariates,
        os.path.join(output_dir, "aparc_lh_rh.rds"),
        "Aparc full data (lh/rh)",
    )
    save_dataframe(
        aseg_with_covariates,
        os.path.join(output_dir, "aseg_lh_rh.rds"),
        "Aseg full data (lh/rh)",
    )
    save_dataframe(
        full_with_covariates,
        os.path.join(output_dir, "full_lh_rh.rds"),
        "full data (lh/rh)",
    )
    save_dataframe(
        aparc_combined_with_covariates,
        os.path.join(output_dir, "aparc.rds"),
        "Aparc full data (combined)",
    )
    # save_dataframe(aseg_combined_with_covariates, os.path.join(output_dir, "aseg.rds"), "Aseg full data (combined)")
    save_dataframe(
        combined_full_with_covariates,
        os.path.join(output_dir, "full.rds"),
        "full data (combined)",
    )


def main():
    """Main function to load and process the data."""
    covariates = load_core_data(CORE_FILE)

    aparc_data = load_and_process_files(APARC_FILES, WAVE_MAPPING)
    aseg_data = load_and_process_files(ASEG_FILES, WAVE_MAPPING, is_aseg=True)
    full_data = pd.merge(aparc_data, aseg_data, on=["idc", "wave"], how="inner")

    process_and_save_with_combined(
        aparc_data, aseg_data, full_data, covariates, OUTPUT_DIR
    )

    process_and_save_subsets(aparc_data, OUTPUT_DIR, "aparc", covariates)
    process_and_save_subsets(full_data, OUTPUT_DIR, "full", covariates)


if __name__ == "__main__":
    main()
