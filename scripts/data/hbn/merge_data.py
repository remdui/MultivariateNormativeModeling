"""Merge and process HBN MRI data from FreeSurfer output files."""

import logging
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import pyreadr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # forces reconfiguration
)

# ----------------------------
# User-defined Parameters
# ----------------------------
DATASET_PREFIX = "hbn"  # Dataset prefix (e.g., "hbn")
ID_FIELD = "EID"  # Identifier column name
COVARIATE_COLUMNS = [ID_FIELD, "Age", "Sex", "site"]

ENABLE_QC = True

# Keywords used to identify measure-related columns
AREA_KEY = "surfarea"  # was "area"
VOLUME_KEY = "vol"  # was "volume"
THICKNESS_KEY = "thickavg"  # was "thickness"

# Keys for hemisphere columns
LH_KEY = "lh_"
RH_KEY = "rh_"

# File names built using the dataset prefix
OUTPUT_DIR = "../../../data"
APARC_FILES = [f"aparc_{DATASET_PREFIX}.rds"]
ASEG_FILES = [f"aseg_{DATASET_PREFIX}.rds"]
CORE_FILE = f"core_{DATASET_PREFIX}.rds"
QC_FILE = f"{DATASET_PREFIX}_qc.rds"

# Features to exclude (if any)
APARC_EXCLUDE_FEATURES = [
    "lh_MeanThickness_thickness",
    "rh_MeanThickness_thickness",
    "lh_WhiteSurfArea_area",
    "rh_WhiteSurfArea_area",
]

# ASEG Features that are available in the GENR dataset
INCLUDED_ASEG_FEATURES = {
    "Brain_Stem",
    "CC_Anterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Mid_Posterior",
    "CC_Posterior",
    "CSF",
    "Fifth_Ventricle",
    "Fourth_Ventricle",
    "Left_Accumbens_area",
    "Left_Amygdala",
    "Left_Caudate",
    "Left_Cerebellum_Cortex",
    "Left_Cerebellum_White_Matter",
    "Left_choroid_plexus",
    "Left_Hippocampus",
    "Left_Inf_Lat_Vent",
    "Left_Lateral_Ventricle",
    "Left_non_WM_hypointensities",
    "Left_Pallidum",
    "Left_Putamen",
    "Left_Thalamus_Proper",
    "Left_VentralDC",
    "Left_vessel",
    "Left_WM_hypointensities",
    "non_WM_hypointensities",
    "Optic_Chiasm",
    "Right_Accumbens_area",
    "Right_Amygdala",
    "Right_Caudate",
    "Right_Cerebellum_Cortex",
    "Right_Cerebellum_White_Matter",
    "Right_choroid_plexus",
    "Right_Hippocampus",
    "Right_Inf_Lat_Vent",
    "Right_Lateral_Ventricle",
    "Right_non_WM_hypointensities",
    "Right_Pallidum",
    "Right_Putamen",
    "Right_Thalamus_Proper",
    "Right_VentralDC",
    "Right_vessel",
    "Right_WM_hypointensities",
    "Third_Ventricle",
    "WM_hypointensities",
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Function Definitions
# ----------------------------
def load_core_data(file_path):
    """Load the core data file and process the covariates."""
    core_data = pyreadr.read_r(file_path)[None]
    covariates = core_data[COVARIATE_COLUMNS]

    # Rename columns to lowercase
    covariates.columns = [
        col if col == ID_FIELD else col.lower() for col in covariates.columns
    ]

    # Convert types using the new lowercase names
    covariates["age"] = covariates["age"].astype(np.float64)
    covariates["sex"] = covariates["sex"].astype(np.float64)
    covariates["site"] = covariates["site"].astype(np.float64)

    logging.info(
        f"Core data loaded with {covariates.shape[0]} rows and columns: {list(covariates.columns)}"
    )

    return covariates


def rename_suffixes(df):
    """
    Rename the suffixes of the columns:

      - _area      -> _surfarea
      - _volume    -> _vol
      - _thickness -> _thickavg
    Also converts column names to lowercase for consistency.
    """
    new_columns = {}
    for col in df.columns:
        new_col = re.sub(r"_area$", "_surfarea", col)
        new_col = re.sub(r"_volume$", "_vol", new_col)
        new_col = re.sub(r"_thickness$", "_thickavg", new_col)
        new_columns[col] = new_col
    df.rename(columns=new_columns, inplace=True)
    return df


def process_aseg_feature_names(df):
    """
    Processes a list of feature names by:

      1. Replacing dots ('.') with underscores ('_').
      2. Renaming:
           - "X3rd.Ventricle" -> "Third_Ventricle"
           - "X4th.Ventricle" -> "Fourth_Ventricle"
           - "X5th.Ventricle" -> "Fifth_Ventricle"
      3. Filtering to only keep allowed features.

    Parameters:
        df (pd.DataFrame): Input DataFrame with raw column names.

    Returns:
        List of processed feature names that are in the allowed subset.
    """
    # Create a mapping from the original column names to the new names
    mapping = {}
    for col in df.columns:
        # Replace dots with underscores
        new_col = col.replace(".", "_")

        # Rename specific ventricle columns if applicable
        if new_col == "X3rd_Ventricle":
            new_col = "Third_Ventricle"
        elif new_col == "X4th_Ventricle":
            new_col = "Fourth_Ventricle"
        elif new_col == "X5th_Ventricle":
            new_col = "Fifth_Ventricle"

        mapping[col] = new_col

    # Rename the DataFrame's columns using the mapping
    df_renamed = df.rename(columns=mapping)

    # Define the special columns that should always be kept
    special_keep = {"row_id"}.union(COVARIATE_COLUMNS)

    # Combine allowed features with the special keep columns
    allowed_set = set(INCLUDED_ASEG_FEATURES).union(special_keep)

    # Filter the DataFrame to only keep columns that are in the allowed set
    cols_to_keep = [col for col in df_renamed.columns if col in allowed_set]
    df_filtered = df_renamed[cols_to_keep]

    return df_filtered


def load_and_process_files(file_list, is_aseg=False):
    """Load and process the FreeSurfer output files."""
    processed_data = []
    for file_path in file_list:
        data = pyreadr.read_r(file_path)[None]
        if is_aseg:
            data = process_aseg_feature_names(data)
            data.rename(
                columns=lambda col: col[:-4] if col.endswith("_vol") else col,
                inplace=True,
            )
        data = data.drop(
            columns=[col for col in APARC_EXCLUDE_FEATURES if col in data.columns],
            errors="ignore",
        )
        data = data.astype(np.float64, errors="ignore")
        data = rename_suffixes(data)
        processed_data.append(data)
    combined_data = pd.concat(processed_data, ignore_index=True).dropna()
    return combined_data


def identify_hemisphere_columns(df):
    """Identify hemisphere-related columns and other columns using LH_KEY and RH_KEY."""
    hemisphere_columns = [
        col for col in df.columns if col.startswith(LH_KEY) or col.startswith(RH_KEY)
    ]
    other_columns = [col for col in df.columns if col not in hemisphere_columns]
    return hemisphere_columns, other_columns


def combine_hemisphere_pairs(df, hemisphere_columns):
    """Combine left and right hemisphere columns into single averaged columns using LH_KEY and RH_KEY."""
    combined_columns = {}
    # Create a mapping from base name to left hemisphere column
    lh_base_names = {
        col[len(LH_KEY) :]: col for col in hemisphere_columns if col.startswith(LH_KEY)
    }
    for base_name, lh_col in lh_base_names.items():
        rh_col = f"{RH_KEY}{base_name}"
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
    merged_data = covariates.merge(data, on=[ID_FIELD], how="inner")
    combined_merged_data = covariates.merge(combined_data, on=[ID_FIELD], how="inner")
    # Save results
    save_dataframe(
        merged_data,
        os.path.join(output_dir, f"{DATASET_PREFIX}_{prefix}_{name}_lh_rh.rds"),
        f"{prefix} {name} (lh/rh)",
    )
    save_dataframe(
        combined_merged_data,
        os.path.join(output_dir, f"{DATASET_PREFIX}_{prefix}_{name}.rds"),
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
        process_subset_and_save(pair_data, covariates, output_dir, prefix, pair_name)


def process_subset_single(covariates, df, other_features, output_dir, prefix, subsets):
    """Process and save the data for each subset."""
    for subset_name, subset_cols in subsets.items():
        subset_features = subset_cols + other_features
        subset_data = df[subset_features].dropna()
        process_subset_and_save(
            subset_data, covariates, output_dir, prefix, subset_name
        )


def create_subsets(df, output_dir, prefix, covariates):
    """Process and save subsets of the data."""
    # Identify features of interest based on the defined keywords
    features_of_interest = [
        col
        for col in df.columns
        if any(
            keyword in col.lower() for keyword in (AREA_KEY, VOLUME_KEY, THICKNESS_KEY)
        )
    ]
    other_features = [col for col in df.columns if col not in features_of_interest]
    subsets = get_subset_features(features_of_interest)
    process_subset_single(covariates, df, other_features, output_dir, prefix, subsets)
    process_subset_pair(covariates, df, other_features, output_dir, prefix, subsets)


def get_subset_features(features_of_interest):
    """Get the subset features for the data using the defined keywords."""
    subsets = {
        AREA_KEY: [col for col in features_of_interest if AREA_KEY in col.lower()],
        VOLUME_KEY: [col for col in features_of_interest if VOLUME_KEY in col.lower()],
        THICKNESS_KEY: [
            col for col in features_of_interest if THICKNESS_KEY in col.lower()
        ],
    }
    return subsets


def process_and_save_with_combined(
    aparc_data, aseg_data, full_data, covariates, output_dir
):
    """Process and save the data with combined hemisphere columns."""
    aparc_combined = combine_hemisphere_columns(aparc_data)
    combined_full_data = combine_hemisphere_columns(full_data)
    aparc_with_covariates = covariates.merge(aparc_data, on=[ID_FIELD], how="inner")
    aseg_with_covariates = covariates.merge(aseg_data, on=[ID_FIELD], how="inner")
    full_with_covariates = covariates.merge(full_data, on=[ID_FIELD], how="inner")
    aparc_combined_with_covariates = covariates.merge(
        aparc_combined, on=[ID_FIELD], how="inner"
    )
    combined_full_with_covariates = covariates.merge(
        combined_full_data, on=[ID_FIELD], how="inner"
    )

    save_dataframe(
        aparc_with_covariates,
        os.path.join(output_dir, f"{DATASET_PREFIX}_aparc_lh_rh.rds"),
        "Aparc full data (lh/rh)",
    )
    save_dataframe(
        aseg_with_covariates,
        os.path.join(output_dir, f"{DATASET_PREFIX}_aseg_lh_rh.rds"),
        "Aseg full data (lh/rh)",
    )
    save_dataframe(
        full_with_covariates,
        os.path.join(output_dir, f"{DATASET_PREFIX}_full_lh_rh.rds"),
        "Full data (lh/rh)",
    )
    save_dataframe(
        aparc_combined_with_covariates,
        os.path.join(output_dir, f"{DATASET_PREFIX}_aparc.rds"),
        "Aparc full data (combined)",
    )
    save_dataframe(
        combined_full_with_covariates,
        os.path.join(output_dir, f"{DATASET_PREFIX}_full.rds"),
        "Full data (combined)",
    )


# ----------------------------
# Main Function
# ----------------------------
def main():
    """Main function to load, filter, and process the data."""
    # Load core, aparc, and aseg data
    covariates = load_core_data(CORE_FILE)
    aparc_data = load_and_process_files(APARC_FILES)
    aseg_data = load_and_process_files(ASEG_FILES, is_aseg=True)

    # Log initial counts before filtering
    logging.info(
        f"Initial counts: covariates: {covariates.shape[0]}, aparc: {aparc_data.shape[0]}, aseg: {aseg_data.shape[0]}"
    )

    if ENABLE_QC:
        # -----------------------------------------------
        # Load QC file and filter out excluded and duplicate EIDs
        # -----------------------------------------------
        qc_data = pyreadr.read_r(QC_FILE)[None]
        logging.info(f"QC data loaded with shape {qc_data.shape}")

        # Exclude rows with exclude == 1 and drop duplicate QC entries based on ID_FIELD
        qc_valid = qc_data[qc_data["exclude"] != 1].drop_duplicates(subset=[ID_FIELD])
        valid_eids = set(qc_valid[ID_FIELD])
        logging.info(f"{len(valid_eids)} valid EIDs after QC filtering")

        # Filter the data frames by valid EIDs and log counts
        covariates_before = covariates.shape[0]
        covariates = covariates[covariates[ID_FIELD].isin(valid_eids)]
        logging.info(
            f"Covariates: Kept {covariates.shape[0]} rows out of {covariates_before} (removed {covariates_before - covariates.shape[0]} rows) after filtering by valid EIDs."
        )

        aparc_before = aparc_data.shape[0]
        aparc_data = aparc_data[aparc_data[ID_FIELD].isin(valid_eids)]
        logging.info(
            f"Aparc: Kept {aparc_data.shape[0]} rows out of {aparc_before} (removed {aparc_before - aparc_data.shape[0]} rows) after filtering by valid EIDs."
        )

        aseg_before = aseg_data.shape[0]
        aseg_data = aseg_data[aseg_data[ID_FIELD].isin(valid_eids)]
        logging.info(
            f"Aseg: Kept {aseg_data.shape[0]} rows out of {aseg_before} (removed {aseg_before - aseg_data.shape[0]} rows) after filtering by valid EIDs."
        )

        # For EIDs flagged with duplicate_removed == 1, remove duplicate rows (keep first occurrence)
        qc_dup_ids = qc_valid.loc[qc_valid["duplicate_removed"] == 1, ID_FIELD].unique()

        def remove_duplicates_for_flagged(df):
            mask = df[ID_FIELD].isin(qc_dup_ids)
            return pd.concat([df[~mask], df[mask].drop_duplicates(subset=[ID_FIELD])])

        covariates_before_dup = covariates.shape[0]
        covariates = remove_duplicates_for_flagged(covariates)
        logging.info(
            f"Covariates after duplicate removal: Kept {covariates.shape[0]} rows out of {covariates_before_dup} (removed {covariates_before_dup - covariates.shape[0]} duplicates)."
        )

        aparc_before_dup = aparc_data.shape[0]
        aparc_data = remove_duplicates_for_flagged(aparc_data)
        logging.info(
            f"Aparc after duplicate removal: Kept {aparc_data.shape[0]} rows out of {aparc_before_dup} (removed {aparc_before_dup - aparc_data.shape[0]} duplicates)."
        )

        aseg_before_dup = aseg_data.shape[0]
        aseg_data = remove_duplicates_for_flagged(aseg_data)
        logging.info(
            f"Aseg after duplicate removal: Kept {aseg_data.shape[0]} rows out of {aseg_before_dup} (removed {aseg_before_dup - aseg_data.shape[0]} duplicates)."
        )
    else:
        logging.info("QC filtering is disabled. Skipping QC filtering steps.")

    # -----------------------------------------------
    # Merge datasets and proceed with processing and saving
    # -----------------------------------------------
    full_data = pd.merge(aparc_data, aseg_data, on=[ID_FIELD], how="inner")
    logging.info(
        f"Full data after merging aparc and aseg has {full_data.shape[0]} rows"
    )

    process_and_save_with_combined(
        aparc_data, aseg_data, full_data, covariates, OUTPUT_DIR
    )
    create_subsets(aparc_data, OUTPUT_DIR, "aparc", covariates)
    create_subsets(full_data, OUTPUT_DIR, "full", covariates)


if __name__ == "__main__":
    main()
