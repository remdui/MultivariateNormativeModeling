"""Create HBN site table from site data files."""

import pandas as pd
import pyreadr


def create_rds_file():
    """Merge text files containing subject IDS into site table."""
    # Define the mapping of sites and file names
    site_files = {
        1: "RUBIC.txt",
        2: "CBIC.txt",
        3: "CUNY.txt",
        4: "SI.txt",
    }

    # Initialize an empty DataFrame to store all the data
    all_data = pd.DataFrame(columns=["EID", "site"])

    # Read each file and assign site codes
    for site_code, file_name in site_files.items():
        try:
            # Read the file, ensure no duplicates, and create a DataFrame with site code
            site_data = pd.read_csv(
                file_name, header=None, names=["EID"]
            ).drop_duplicates()
            site_data["site"] = site_code

            # Append to the master DataFrame
            all_data = pd.concat([all_data, site_data], ignore_index=True)
        except FileNotFoundError:
            print(f"File not found: {file_name}. Skipping...")

    # Ensure the 'site' column is of integer type
    all_data["site"] = all_data["site"].astype(int)

    # Drop duplicates across all sites and ensure EID is unique
    all_data = all_data.drop_duplicates(subset="EID").reset_index(drop=True)

    # Write to an RDS file
    pyreadr.write_rds("hbn_sites.rds", all_data)
    print("RDS file 'hbn_sites.rds' created successfully.")


if __name__ == "__main__":
    create_rds_file()
