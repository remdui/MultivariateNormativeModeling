"""Merge GENR MRI data from different waves and save as RDS files."""

import os

import numpy as np
import pandas as pd
import pyreadr


def combine_hemisphere_columns(df):
    """Combine columns with 'lh_' and 'rh_' prefixes by summing and averaging."""
    lh_columns = [col for col in df.columns if col.startswith("lh_")]
    rh_columns = [col for col in df.columns if col.startswith("rh_")]

    combined_columns = {}
    for lh_col in lh_columns:
        name = lh_col[3:]  # Remove 'lh_' prefix
        rh_col = f"rh_{name}"  # Create corresponding 'rh_' column name
        if rh_col in rh_columns:
            # Calculate the average (mean) of lh_ and rh_
            mean_lh_rh = df[[lh_col, rh_col]].mean(axis=1).astype(np.float64)
            combined_columns[name] = mean_lh_rh

    # Drop all the lh_ and rh_ columns that are paired
    drop_cols = tuple(
        col for pair in combined_columns for col in (f"lh_{pair}", f"rh_{pair}")
    )
    df = df.drop(columns=drop_cols)

    # Add all combined columns at once
    combined_df = pd.DataFrame(combined_columns)
    # Concatenate along columns
    df = pd.concat([df, combined_df], axis=1)

    return df


# Paths to input files and corresponding age fields
aparc_files = [
    "f05_freesurfer_v6_24june2021_aparc_stats_pull18Aug2021_F.rds",
    "f09_freesurfer_v6_09dec2016_aparc_stats_pull06june2017.rds",
    "f13_freesurfer_v6_14oct2020_aparc_stats_pull23Nov2020.rds",
]
age_fields = ["age_child_mri_f05", "age_child_mri_f09", "age_child_mri_f13"]

core_file = "genr_mri_core_data_20231204.rds"

output_dir = "../../../data"
os.makedirs(output_dir, exist_ok=True)

core_data = pyreadr.read_r(core_file)[None]
core_data = core_data[["idc"] + age_fields]

wave_mapping = {"f05": 0, "f09": 1, "f13": 2}
merged_aparc_data = []

for aparc_file, age_field in zip(aparc_files, age_fields):
    wave_label = os.path.basename(aparc_file)[:3]
    aparc_data = pyreadr.read_r(aparc_file)[None]

    nan_rows = aparc_data[aparc_data.isna().any(axis=1)]
    print(f"Rows containing NaN values in {wave_label}:")
    print(nan_rows.index.tolist())

    aparc_data.rename(
        columns=lambda col: col.rstrip("_f09").rstrip("_f05").rstrip("_f13"),
        inplace=True,
    )

    core_subset = core_data[["idc", age_field]].rename(columns={age_field: "age"})
    merged_data = aparc_data.merge(core_subset, on="idc", how="left")
    merged_data["wave"] = wave_mapping[wave_label]
    merged_aparc_data.append(merged_data)

combined_data = pd.concat(merged_aparc_data, ignore_index=True)

# Instead of inplace reset, do it and reassign
combined_data = combined_data.reset_index().rename(columns={"index": "row_id"})

combined_data = combine_hemisphere_columns(combined_data)

if "MeanThickness" in combined_data.columns:
    combined_data = combined_data.drop(columns=["MeanThickness"], errors="ignore")

combined_data = combined_data.dropna()

all_output_file = os.path.join(output_dir, "gen_r_aparc.rds")
pyreadr.write_rds(all_output_file, combined_data)
print("All data saved to:", all_output_file)

id_cols = ["idc", "age", "row_id", "wave"]

surfarea_cols = [
    col for col in combined_data.columns if "surfarea" in col.lower() or col in id_cols
]
surfarea_data = combined_data[surfarea_cols]
surfarea_output_file = os.path.join(output_dir, "gen_r_aparc_surfarea.rds")
pyreadr.write_rds(surfarea_output_file, surfarea_data)
print("Surfarea subset saved to:", surfarea_output_file)
print("Surfarea columns:", surfarea_data.columns.tolist())

vol_cols = [
    col for col in combined_data.columns if "vol" in col.lower() or col in id_cols
]
vol_data = combined_data[vol_cols]
vol_output_file = os.path.join(output_dir, "gen_r_aparc_vol.rds")
pyreadr.write_rds(vol_output_file, vol_data)
print("Vol subset saved to:", vol_output_file)
print("Vol columns:", vol_data.columns.tolist())

thickavg_cols = [
    col for col in combined_data.columns if "thickavg" in col.lower() or col in id_cols
]
thickavg_data = combined_data[thickavg_cols]
thickavg_output_file = os.path.join(output_dir, "gen_r_aparc_thickavg.rds")
pyreadr.write_rds(thickavg_output_file, thickavg_data)
print("Thickavg subset saved to:", thickavg_output_file)
print("Thickavg columns:", thickavg_data.columns.tolist())

print("All data columns:", combined_data.columns.tolist())
print(combined_data.head())
print("All data shape:", combined_data.shape)
