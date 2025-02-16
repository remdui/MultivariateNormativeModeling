"""Merge and process GENR MRI data from FreeSurfer output files with QC filtering.

that checks both idc and wave fields AFTER merging.

This script loads core, aparc, and aseg data files, then merges them with covariates.
After merging, a QC function is applied that filters out rows where the QC file flags exclude==1
and removes duplicate entries (by idc and wave) flagged with duplicate_removed==1.
Finally, the processed data are saved.
"""

import logging
import os
from itertools import combinations

import numpy as np
import pandas as pd
import pyreadr

# ----------------------------
# User-defined Parameters
# ----------------------------
OUTPUT_DIR = "../../../data"
WAVE_MAPPING = {"f05": 0, "f09": 1, "f13": 2}

# File names for FreeSurfer output
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

# QC file: must contain columns "idc", "wave", "exclude", and "duplicate_removed"
QC_FILE = "genr_qc.rds"

# Features to exclude (if any)
APARC_EXCLUDE_FEATURES = ["lh_MeanThickness", "rh_MeanThickness"]
ASEG_EXCLUDE_FEATURES = []

# Toggle QC filtering on or off (QC filtering is now applied after merging)
ENABLE_QC = True  # Set to False to disable QC filtering

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ----------------------------
# QC Filtering Function (applied after merging)
# ----------------------------
def apply_qc_filter(merged_df, qc_data):
    """
    Apply QC filtering to a merged dataframe.

    This function:
      - Ensures that 'idc' is of type str and 'wave' is numeric.
      - Merges in the QC columns ("exclude", "duplicate_removed") based on idc and wave.
      - Logs the number of rows missing QC info and the number flagged for exclusion.
      - Drops rows where exclude == 1.
      - Removes duplicate rows (by idc and wave) flagged with duplicate_removed == 1 (keeping the first occurrence).
      - Drops the QC columns before returning.
    """
    qc_data["idc"] = qc_data["idc"].astype(np.float64)
    qc_data["wave"] = qc_data["wave"].astype(np.float64)
    qc_data["exclude"] = qc_data["exclude"].astype(np.int32)

    # Merge QC information into the merged dataframe.
    merged_with_qc = merged_df.merge(
        qc_data[["idc", "wave", "exclude", "duplicate_removed"]],
        on=["idc", "wave"],
        how="left",
    )

    # Filter rows where exclude == 1 from the QC data.
    excluded_rows = qc_data[qc_data["exclude"] == 1]

    # Check if any of the excluded idc values are present in merged_df.
    common_ids = set(merged_df["idc"]).intersection(set(excluded_rows["idc"]))
    if common_ids:
        print(
            "The following idc values from QC excluded rows are present in the merged dataframe:"
        )
        print(list(common_ids))
    else:
        print(
            "No idc values from QC excluded rows are present in the merged dataframe."
        )

    # Exclude rows flagged for exclusion.
    filtered = merged_with_qc[merged_with_qc["exclude"] != 1].copy()

    # Remove duplicate rows (if any) based on idc and wave.
    filtered = filtered.drop_duplicates(subset=["idc", "wave"])

    # Optionally, drop the QC columns.
    filtered = filtered.drop(columns=["exclude", "duplicate_removed"], errors="ignore")

    print(
        f"After QC filtering, {filtered.shape[0]} rows remain out of {merged_df.shape[0]} rows"
    )
    return filtered


# ----------------------------
# Function Definitions (Imaging Data Processing)
# ----------------------------
def load_core_data(file_path):
    """Load the core data file and process the covariates."""
    core_data = pyreadr.read_r(file_path)[None]
    # Melt age columns from wide to long format.
    covariates = core_data[
        ["idc", "age_child_mri_f05", "age_child_mri_f09", "age_child_mri_f13"]
    ]
    covariates = pd.melt(
        covariates, id_vars=["idc"], var_name="wave", value_name="age"
    ).dropna()
    # Map wave labels (last three characters) to numeric values and cast columns to float.
    covariates["wave"] = covariates["wave"].str[-3:].map(WAVE_MAPPING)
    covariates["age"] = covariates["age"].astype(np.float64)
    covariates["wave"] = covariates["wave"].astype(np.float64)
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
        # Remove wave-specific suffixes from column names.
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
        # Append a column for wave (using the provided mapping).
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
    # Create a mapping from base name to left hemisphere column.
    lh_base_names = {
        col[len("lh_") :]: col for col in hemisphere_columns if col.startswith("lh_")
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


# ----------------------------
# Functions for Subset Processing (after merging with covariates)
# ----------------------------
def process_subset_and_save(data, covariates, qc_data, output_dir, prefix, name):
    """
    Combine hemisphere columns, merge with covariates, apply QC filtering,.

    and save both the original (lh/rh) and combined versions.
    """
    combined_data = combine_hemisphere_columns(data)
    merged_data = covariates.merge(data, on=["idc", "wave"], how="inner")
    merged_data = apply_qc_filter(merged_data, qc_data)
    combined_merged_data = covariates.merge(
        combined_data, on=["idc", "wave"], how="inner"
    )
    combined_merged_data = apply_qc_filter(combined_merged_data, qc_data)
    # Save results.
    save_dataframe(
        merged_data,
        os.path.join(output_dir, f"genr_{prefix}_{name}_lh_rh.rds"),
        f"{prefix} {name} (lh/rh)",
    )
    save_dataframe(
        combined_merged_data,
        os.path.join(output_dir, f"genr_{prefix}_{name}.rds"),
        f"{prefix} {name} (combined)",
    )


def process_subset_pair(
    covariates, df, other_features, qc_data, output_dir, prefix, subsets
):
    """Process and save the data for each pair of subsets."""
    subset_names = list(subsets.keys())
    for pair in combinations(subset_names, 2):
        pair_name = "_and_".join(pair)
        pair_columns = list(set(subsets[pair[0]] + subsets[pair[1]]))
        pair_features = pair_columns + other_features
        pair_data = df[pair_features].dropna()
        process_subset_and_save(
            pair_data, covariates, qc_data, output_dir, prefix, pair_name
        )


def process_subset_single(
    covariates, df, other_features, qc_data, output_dir, prefix, subsets
):
    """Process and save the data for each subset."""
    for subset_name, subset_cols in subsets.items():
        subset_features = subset_cols + other_features
        subset_data = df[subset_features].dropna()
        process_subset_and_save(
            subset_data, covariates, qc_data, output_dir, prefix, subset_name
        )


def get_subset_features(features_of_interest):
    """Get the subset features for the data."""
    subsets = {
        "surfarea": [col for col in features_of_interest if "surfarea" in col.lower()],
        "vol": [col for col in features_of_interest if "vol" in col.lower()],
        "thickavg": [col for col in features_of_interest if "thickavg" in col.lower()],
    }
    return subsets


def create_subsets(df, output_dir, prefix, covariates, qc_data):
    """Process and save subsets of the data."""
    features_of_interest = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in ("surfarea", "vol", "thickavg"))
    ]
    other_features = [col for col in df.columns if col not in features_of_interest]
    subsets = get_subset_features(features_of_interest)
    process_subset_single(
        covariates, df, other_features, qc_data, output_dir, prefix, subsets
    )
    process_subset_pair(
        covariates, df, other_features, qc_data, output_dir, prefix, subsets
    )


def process_and_save_with_combined(
    aparc_data, aseg_data, full_data, covariates, qc_data, output_dir
):
    """Process and save the data with combined hemisphere columns, applying QC filtering after merging."""
    aparc_combined = combine_hemisphere_columns(aparc_data)
    combined_full_data = combine_hemisphere_columns(full_data)

    aparc_with_covariates = covariates.merge(
        aparc_data, on=["idc", "wave"], how="inner"
    )
    aparc_with_covariates = apply_qc_filter(aparc_with_covariates, qc_data)

    aseg_with_covariates = covariates.merge(aseg_data, on=["idc", "wave"], how="inner")
    aseg_with_covariates = apply_qc_filter(aseg_with_covariates, qc_data)

    full_with_covariates = covariates.merge(full_data, on=["idc", "wave"], how="inner")
    full_with_covariates = apply_qc_filter(full_with_covariates, qc_data)

    aparc_combined_with_covariates = covariates.merge(
        aparc_combined, on=["idc", "wave"], how="inner"
    )
    aparc_combined_with_covariates = apply_qc_filter(
        aparc_combined_with_covariates, qc_data
    )

    combined_full_with_covariates = covariates.merge(
        combined_full_data, on=["idc", "wave"], how="inner"
    )
    combined_full_with_covariates = apply_qc_filter(
        combined_full_with_covariates, qc_data
    )

    save_dataframe(
        aparc_with_covariates,
        os.path.join(output_dir, "genr_aparc_lh_rh.rds"),
        "Aparc full data (lh/rh)",
    )
    save_dataframe(
        aseg_with_covariates,
        os.path.join(output_dir, "genr_aseg_lh_rh.rds"),
        "Aseg full data (lh/rh)",
    )
    save_dataframe(
        full_with_covariates,
        os.path.join(output_dir, "genr_full_lh_rh.rds"),
        "Full data (lh/rh)",
    )
    save_dataframe(
        aparc_combined_with_covariates,
        os.path.join(output_dir, "genr_aparc.rds"),
        "Aparc full data (combined)",
    )
    save_dataframe(
        combined_full_with_covariates,
        os.path.join(output_dir, "genr_full.rds"),
        "Full data (combined)",
    )


# ----------------------------
# Main Function
# ----------------------------
def main():
    """Main function to load, merge, and process the data with QC filtering applied after covariate merging."""
    # Load core, aparc, and aseg data.
    covariates = load_core_data(CORE_FILE)
    aparc_data = load_and_process_files(APARC_FILES, WAVE_MAPPING)
    aseg_data = load_and_process_files(ASEG_FILES, WAVE_MAPPING, is_aseg=True)

    logging.info(
        f"Initial counts: covariates: {covariates.shape[0]}, aparc: {aparc_data.shape[0]}, aseg: {aseg_data.shape[0]}"
    )

    # Load QC data (once) and ensure its key columns have the correct types.
    qc_data = pyreadr.read_r(QC_FILE)[None]

    # Merge aparc and aseg to form the full dataset.
    full_data = pd.merge(aparc_data, aseg_data, on=["idc", "wave"], how="inner")
    logging.info(
        f"Full data after merging aparc and aseg has {full_data.shape[0]} rows"
    )

    # Process and save the combined datasets.
    process_and_save_with_combined(
        aparc_data, aseg_data, full_data, covariates, qc_data, OUTPUT_DIR
    )
    create_subsets(aparc_data, OUTPUT_DIR, "aparc", covariates, qc_data)
    create_subsets(full_data, OUTPUT_DIR, "full", covariates, qc_data)


if __name__ == "__main__":
    main()
