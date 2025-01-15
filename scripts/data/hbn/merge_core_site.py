"""Merge HBN Core data with HBN Site data."""

import pyreadr


def merge_core_site():
    """Function to merge core and site datasets."""
    # Load core_hbn.rds
    core_data = pyreadr.read_r("core_hbn.rds")[None]

    # Load hbn_sites.rds
    site_data = pyreadr.read_r("hbn_sites.rds")[None]

    # Ensure both DataFrames have the correct columns
    if (
        "EID" not in core_data.columns
        or "EID" not in site_data.columns
        or "site" not in site_data.columns
    ):
        raise ValueError("Missing required columns in the input RDS files.")

    # Merge the dataframes, using left join to keep all rows in core_data
    merged_data = core_data.merge(site_data, on="EID", how="left")

    # Fill NaN values in the 'site' column with -1 for missing sites
    merged_data["site"] = merged_data["site"].fillna(-1).astype(int)

    # Write the merged data back to an RDS file
    pyreadr.write_rds("core_hbn.rds", merged_data)
    print("RDS file 'merged_core_hbn.rds' created successfully.")


if __name__ == "__main__":
    merge_core_site()
