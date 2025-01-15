"""Merge and process HBN MRI data from FreeSurfer output files."""

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
APARC_FILES = [
    "aparc_hbn.rds",
]
ASEG_FILES = [
    "aseg_hbn.rds",
]
CORE_FILE = "core_hbn.rds"

# Features to exclude
APARC_EXCLUDE_FEATURES = []
ASEG_EXCLUDE_FEATURES = []

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_core_data(file_path):
    """Load the core data file and process the covariates."""
    core_data = pyreadr.read_r(file_path)[None]
    covariates = core_data[["EID", "Age", "Sex", "site"]]
    covariates["Age"] = covariates["Age"].astype(np.float64)
    covariates["Sex"] = covariates["Sex"].astype(np.float64)
    covariates["site"] = covariates["site"].astype(np.float64)
    logging.info(
        f"Core data loaded with {covariates.shape[0]} rows and columns: {list(covariates.columns)}"
    )
    return covariates


def load_and_process_files(file_list, is_aseg=False):
    """Load and process the FreeSurfer output files."""
    processed_data = []
    exclude_features = ASEG_EXCLUDE_FEATURES if is_aseg else APARC_EXCLUDE_FEATURES
    for file_path in file_list:
        data = pyreadr.read_r(file_path)[None]

        if is_aseg:
            data.rename(
                columns=lambda col: col[:-4] if col.endswith("_vol") else col,
                inplace=True,
            )
        data = data.drop(
            columns=[col for col in exclude_features if col in data.columns],
            errors="ignore",
        )

        data = data.astype(np.float64, errors="ignore")
        processed_data.append(data)
    combined_data = pd.concat(processed_data, ignore_index=True).dropna()
    return combined_data


def identify_hemisphere_columns(df):
    """Identify hemisphere-related columns and other columns."""
    hemisphere_columns = [
        col for col in df.columns if col.startswith("lh_") or col.startswith("rh_")
    ]
    other_columns = [col for col in df.columns if col not in hemisphere_columns]
    return hemisphere_columns, other_columns


def combine_hemisphere_pairs(df, hemisphere_columns):
    """Combine left and right hemisphere columns into single averaged columns."""
    combined_columns = {}
    # Extract the base name from columns that have 'lh_'
    lh_base_names = {
        c[len("lh_") :]: f"lh_{c[len('lh_'): ]}"
        for c in hemisphere_columns
        if c.startswith("lh_")
    }

    for base_name, lh_col in lh_base_names.items():
        rh_col = f"rh_{base_name}"
        if lh_col in df.columns and rh_col in df.columns:
            combined_columns[base_name] = (
                df[[lh_col, rh_col]].mean(axis=1).astype(np.float64)
            )
    return pd.DataFrame(combined_columns, index=df.index)


def combine_hemisphere_columns(df):
    """Combine the left and right hemisphere columns into averaged columns."""
    hemisphere_columns, other_columns = identify_hemisphere_columns(df)
    combined_df = combine_hemisphere_pairs(df, hemisphere_columns)
    return pd.concat([df[other_columns], combined_df], axis=1)


def save_dataframe(df, file_path, description):
    """Save the dataframe to an RDS file."""
    df = df.reset_index().rename(columns={"index": "row_id"})
    pyreadr.write_rds(file_path, df)
    logging.info(
        f"{description} saved to {file_path} with {df.shape[0]} rows and columns: {list(df.columns)}\n\n"
    )


def process_subset_and_save(data, covariates, output_dir, prefix, name):
    """Combine hemisphere columns, merge with covariates, and save both lh/rh and combined data."""
    combined_data = combine_hemisphere_columns(data)
    merged_data = covariates.merge(data, on=["EID"], how="inner")
    combined_merged_data = covariates.merge(combined_data, on=["EID"], how="inner")

    # Save results
    save_dataframe(
        merged_data,
        os.path.join(output_dir, f"hbn_{prefix}_{name}_lh_rh.rds"),
        f"{prefix} {name} (lh/rh)",
    )
    save_dataframe(
        combined_merged_data,
        os.path.join(output_dir, f"hbn_{prefix}_{name}.rds"),
        f"{prefix} {name} (combined)",
    )


def process_subset_pair(covariates, df, other_features, output_dir, prefix, subsets):
    """Process and save the data for each pair of subsets."""
    subset_names = list(subsets.keys())
    for pair in combinations(subset_names, 2):
        pair_name = "_and_".join(pair)
        pair_columns = list(set(subsets[pair[0]] + subsets[pair[1]]))
        pair_features = pair_columns + other_features
        pair_data = df[pair_features].dropna()

        # Reuse the helper function
        process_subset_and_save(pair_data, covariates, output_dir, prefix, pair_name)


def process_subset_single(covariates, df, other_features, output_dir, prefix, subsets):
    """Process and save the data for each subset."""
    for subset_name, subset_cols in subsets.items():
        # Keep only the features for the current subset
        subset_features = subset_cols + other_features
        subset_data = df[subset_features].dropna()

        # Reuse the helper function
        process_subset_and_save(
            subset_data, covariates, output_dir, prefix, subset_name
        )


def create_subsets(df, output_dir, prefix, covariates):
    """Process and save subsets of the data."""
    features_of_interest = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in ("surfarea", "vol", "thickavg"))
    ]
    other_features = [col for col in df.columns if col not in features_of_interest]

    # Subset definitions
    subsets = get_subset_features(features_of_interest)

    # Process and save the data for each subset
    process_subset_single(covariates, df, other_features, output_dir, prefix, subsets)
    process_subset_pair(covariates, df, other_features, output_dir, prefix, subsets)


def get_subset_features(features_of_interest):
    """Get the subset features for the data."""
    subsets = {
        "surfarea": [col for col in features_of_interest if "surfarea" in col.lower()],
        "vol": [col for col in features_of_interest if "vol" in col.lower()],
        "thickavg": [col for col in features_of_interest if "thickavg" in col.lower()],
    }
    return subsets


def process_and_save_with_combined(
    aparc_data, aseg_data, full_data, covariates, output_dir
):
    """Process and save the data with combined hemisphere columns."""
    aparc_combined = combine_hemisphere_columns(aparc_data)
    # aseg_combined = combine_hemisphere_columns(aseg_data)
    combined_full_data = combine_hemisphere_columns(full_data)

    aparc_with_covariates = covariates.merge(aparc_data, on=["EID"], how="inner")
    aseg_with_covariates = covariates.merge(aseg_data, on=["EID"], how="inner")
    full_with_covariates = covariates.merge(full_data, on=["EID"], how="inner")

    aparc_combined_with_covariates = covariates.merge(
        aparc_combined, on=["EID"], how="inner"
    )
    # aseg_combined_with_covariates = covariates.merge(aseg_combined, on=["EID"], how="inner")
    combined_full_with_covariates = covariates.merge(
        combined_full_data, on=["EID"], how="inner"
    )

    save_dataframe(
        aparc_with_covariates,
        os.path.join(output_dir, "hbn_aparc_lh_rh.rds"),
        "Aparc full data (lh/rh)",
    )
    save_dataframe(
        aseg_with_covariates,
        os.path.join(output_dir, "hbn_aseg_lh_rh.rds"),
        "Aseg full data (lh/rh)",
    )
    save_dataframe(
        full_with_covariates,
        os.path.join(output_dir, "hbn_full_lh_rh.rds"),
        "full data (lh/rh)",
    )
    save_dataframe(
        aparc_combined_with_covariates,
        os.path.join(output_dir, "hbn_aparc.rds"),
        "Aparc full data (combined)",
    )
    # save_dataframe(aseg_combined_with_covariates, os.path.join(output_dir, "aseg.rds"), "Aseg full data (combined)")
    save_dataframe(
        combined_full_with_covariates,
        os.path.join(output_dir, "hbn_full.rds"),
        "full data (combined)",
    )


def main():
    """Main function to load and process the data."""
    covariates = load_core_data(CORE_FILE)

    aparc_data = load_and_process_files(APARC_FILES)
    aseg_data = load_and_process_files(ASEG_FILES, is_aseg=True)
    full_data = pd.merge(aparc_data, aseg_data, on=["EID"], how="inner")

    process_and_save_with_combined(
        aparc_data, aseg_data, full_data, covariates, OUTPUT_DIR
    )

    create_subsets(aparc_data, OUTPUT_DIR, "aparc", covariates)
    create_subsets(full_data, OUTPUT_DIR, "full", covariates)


if __name__ == "__main__":
    main()
