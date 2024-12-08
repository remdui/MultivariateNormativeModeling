"""Merge and process GENR aparc data from multiple waves."""

import os

import pandas as pd
import pyreadr


# Summing and averaging lh_ and rh_ pairs
def combine_hemisphere_columns(df):
    """Combine columns with 'lh_' and 'rh_' prefixes by summing and averaging."""
    # Identify columns with 'lh_' and 'rh_' prefixes
    lh_columns = [col for col in df.columns if col.startswith("lh_")]
    rh_columns = [col for col in df.columns if col.startswith("rh_")]

    # Dictionary to store column mapping for combining
    combined_columns = {}

    # Find matching pairs
    for lh_col in lh_columns:
        name = lh_col[3:]  # Remove 'lh_' prefix
        rh_col = f"rh_{name}"  # Create corresponding 'rh_' column name
        if rh_col in rh_columns:
            combined_columns[name] = (lh_col, rh_col)

    # Replace the pairs with combined columns
    for name, (lh_col, rh_col) in combined_columns.items():
        # Calculate sum and average
        df[f"{name}"] = df[[lh_col, rh_col]].mean(axis=1)

        # Drop the original lh_ and rh_ columns
        df.drop(columns=[lh_col, rh_col], inplace=True)

    return df


# Paths to input files and corresponding age fields
aparc_files = [
    "f05_freesurfer_v6_24june2021_aparc_stats_pull18Aug2021_F.rds",
    "f09_freesurfer_v6_09dec2016_aparc_stats_pull06june2017.rds",
    "f13_freesurfer_v6_14oct2020_aparc_stats_pull23Nov2020.rds",
]
age_fields = ["age_child_mri_f05", "age_child_mri_f09", "age_child_mri_f13"]
core_file = "genr_mri_core_data_20231204.rds"

# Output directory
output_dir = "../../../data"
os.makedirs(output_dir, exist_ok=True)

# Load core data
core_data = pyreadr.read_r(core_file)[None]  # Load as pandas DataFrame
core_data = core_data[["idc"] + age_fields]  # Keep idc and all age fields

# Initialize an empty list to store merged aparc data
merged_aparc_data = []

# Process each aparc file and corresponding age field
for aparc_file, age_field in zip(aparc_files, age_fields):
    # Load aparc data
    aparc_data = pyreadr.read_r(aparc_file)[None]

    # Remove suffixes from aparc file column names
    aparc_data.rename(
        columns=lambda col: col.rstrip("_f09").rstrip("_f05").rstrip("_f13"),
        inplace=True,
    )

    # Filter core data to include only idc and the relevant age field
    core_subset = core_data[["idc", age_field]].rename(columns={age_field: "age"})

    # Merge with core data on idc
    merged_data = aparc_data.merge(core_subset, on="idc", how="left")

    # Append the merged data to the list
    merged_aparc_data.append(merged_data)

# Combine all waves into one DataFrame
combined_data = pd.concat(merged_aparc_data, ignore_index=True)

# Add a unique row ID
combined_data.reset_index(inplace=True)
combined_data.rename(columns={"index": "row_id"}, inplace=True)

# Apply the function to combined data
combined_data = combine_hemisphere_columns(combined_data)

# Save the combined data as an RDS file in the output directory
output_file = os.path.join(output_dir, "gen_r_aparc_wave_f05_f09_f13.rds")
pyreadr.write_rds(output_file, combined_data)

print(combined_data.head())
print(combined_data.shape)
print(combined_data.columns)

print(f"Combined data saved to: {output_file}")
