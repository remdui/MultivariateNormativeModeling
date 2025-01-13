#!/bin/bash

# Step 1: Load necessary modules
echo "Step 1: Loading required modules"
module load 2022
module load FreeSurfer/7.3.2-centos8_x86_64
module load R/4.2.1-foss-2022a

# Step 2: Set the working directory
WORKING_DIR=$(pwd)  # Get the current directory
echo "Step 2: Setting working directory to $WORKING_DIR"

# Step 3: Check for required CSV files
echo "Step 3: Checking for HBN_R*_Pheno.csv files in $WORKING_DIR"
FILE_PATTERN="HBN_R*_Pheno.csv"
file_list=$(ls $FILE_PATTERN 2>/dev/null)

if [[ -z "$file_list" ]]; then
    echo "Error: No files matching pattern '$FILE_PATTERN' found in $WORKING_DIR"
    exit 1
fi

echo "Found files:"
echo "$file_list"

# Step 4: Import, merge, and save files in R
echo "Step 4: Importing files, merging on EID, and saving as RDS"
Rscript -e "
# Step 4.1: Set working directory
setwd('$WORKING_DIR')

# Step 4.2: List all matching files
file_list <- list.files(pattern = '^HBN_R[0-9]+_Pheno\\\\.csv$')

# Step 4.3: Function to read and validate files
read_pheno_file <- function(file) {
  data <- read.csv(file, stringsAsFactors = FALSE)
  if (!'EID' %in% colnames(data)) {
    stop(paste('EID column not found in file:', file))
  }
  return(data)
}

# Step 4.4: Read all files into a list of data frames
pheno_data <- lapply(file_list, read_pheno_file)

# Step 4.5: Combine all data frames by row-binding
combined_data <- do.call(rbind, pheno_data)

# Step 4.6: Remove duplicate rows based on EID
final_data <- combined_data[!duplicated(combined_data$EID), ]

# Step 4.7: Save the combined data as an RDS file
saveRDS(final_data, 'core_hbn.rds')

cat('Files combined, duplicates removed, and saved as core_hbn.rds successfully.\\n')
"


# Step 5: Completion message
echo "Step 5: All steps completed successfully!"
