"""Explore the structure of the MRI data files."""

import pandas as pd
import pyreadr

# Paths to the wave data files
aparc_files = [
    "f05_freesurfer_v6_24june2021_aparc_stats_pull18Aug2021_F.rds",
    "f09_freesurfer_v6_09dec2016_aparc_stats_pull06june2017.rds",
    "f13_freesurfer_v6_14oct2020_aparc_stats_pull23Nov2020.rds",
]

# Explore and print the structure of each file
for aparc_file in aparc_files:
    print(f"Exploring file: {aparc_file}")

    data = pyreadr.read_r(aparc_file)[None]  # Load as pandas DataFrame

    pd.set_option("display.max_columns", None)

    # Print the shape of the data
    print(f"Shape of {aparc_file}: {data.shape}")

    # Print column names
    print(f"Columns in {aparc_file}:")
    print(data.columns.tolist())
    print("Number of columns:", len(data.columns))
    print("\n")

    # Detect and handle 'object' columns
    object_cols = data.select_dtypes(include=["object"])
    print(f"Object columns detected: {object_cols.columns.tolist()}")
    print("Number of object columns:", len(object_cols.columns))
    print("\n")

    # Select the first 10 object columns
    selected_object_cols = object_cols.columns[:10]

    # Print the first few rows of these object columns (before conversion)
    print("First few rows of selected object columns (before conversion):")
    print(data[selected_object_cols].head())
    print("\n")

    # Try to convert object columns to numeric where possible
    print("Converting object columns to numeric")
    for col in object_cols.columns:
        try:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        except ValueError as e:
            print(f"Error converting column {col} to numeric: {e}")

    # Print the first few rows of the same columns (after conversion)
    print("First few rows of selected object columns (after conversion):")
    print(data[selected_object_cols].head())
    print("\n")

    # Detect and handle 'object' columns after conversion
    converted_object_cols = data.select_dtypes(include=["object"])
    print(
        f"Remaining object columns after conversion: {converted_object_cols.columns.tolist()}"
    )
    print("Number of remaining object columns:", len(converted_object_cols.columns))
    print("\n")

    print("\n" + "-" * 40 + "\n")
