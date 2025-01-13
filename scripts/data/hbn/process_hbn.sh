#!/bin/bash

# Step 1: Load necessary modules
echo "Step 1: Loading required modules"
module load 2022
module load FreeSurfer/7.3.2-centos8_x86_64
module load R/4.2.1-foss-2022a

# Step 2: Set the environment variable for SUBJECTS_FOLDER
SUBJECTS_FOLDER=$(pwd)
export SUBJECTS_DIR=$SUBJECTS_FOLDER

echo "Step 2: Set SUBJECTS_FOLDER to $SUBJECTS_FOLDER"

# Step 3: Loop over .tar.gz files
DATA_DIR="$HOME/mtproject/temp"  # Use $HOME to expand to the full path
echo "Step 3: Processing .tar.gz files from $DATA_DIR"

# Check if DATA_DIR exists and is not empty
if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: Data directory $DATA_DIR does not exist."
    exit 1
fi

found_files=false
for tarfile in "$DATA_DIR"/sub-*_acq-HCP.tar.gz; do
    if [[ -e "$tarfile" ]]; then
        found_files=true
        filename=$(basename "$tarfile")
        subject_name=$(echo "$filename" | cut -d_ -f1)  # Extract subject name

        echo -e "\nCopying $filename to $SUBJECTS_FOLDER"
        cp "$tarfile" "$SUBJECTS_FOLDER"

        echo "Unpacking $filename for 'aseg.stats' and '?h.aparc.stats'"
        tar -xzf "$SUBJECTS_FOLDER/$filename" --wildcards --no-anchored 'aseg.stats' '?h.aparc.stats'

        echo "Removing $filename after unpacking"
        rm "$SUBJECTS_FOLDER/$filename"
    fi
done

if [[ "$found_files" = false ]]; then
    echo "No .tar.gz files found in $DATA_DIR. Exiting..."
    exit 1
fi

# Step 4: Run asegstats2table for aseg
echo "Step 4: Creating aseg_hbn.csv from unpacked subjects"

subject_list=$(ls -d sub-* 2>/dev/null)
if [[ -z "$subject_list" ]]; then
    echo "Error: No subject directories found. Exiting..."
    exit 1
fi

asegstats2table --subjects $subject_list --tablefile "$SUBJECTS_FOLDER/aseg_hbn.csv"

# Step 4.1: Run aparcstats2table for lh and rh separately with different measures
echo "Creating separate lh and rh aparc files with volume, area, and thickness measures"
for measure in volume area thickness; do
    lh_file="$SUBJECTS_FOLDER/aparc_lh_${measure}_hbn.csv"
    rh_file="$SUBJECTS_FOLDER/aparc_rh_${measure}_hbn.csv"

    echo "Generating aparcstats2table for lh with measure=$measure"
    aparcstats2table --subjects $subject_list --hemi lh --measure $measure --tablefile "$lh_file"

    echo "Generating aparcstats2table for rh with measure=$measure"
    aparcstats2table --subjects $subject_list --hemi rh --measure $measure --tablefile "$rh_file"
done

# Step 5: Clean up all unzipped folders
echo "Step 5: Cleaning up unzipped subject folders"
for folder in sub-*; do
    echo "Removing folder $folder"
    rm -rf "$folder"
done

# Step 6: Convert TSV to RDS in R
# Merge all aparc files into a single table
echo "Step 6: Converting aseg and aparc TSVs to RDS and merging aparc files using R"
Rscript -e "
# Load aseg data
aseg_data <- read.delim('$SUBJECTS_FOLDER/aseg_hbn.csv', sep = '\t', header = TRUE, skip = 0)
colnames(aseg_data) <- make.names(colnames(aseg_data))
colnames(aseg_data)[1] <- 'EID'
aseg_data\$EID <- gsub('^sub-|_acq-HCP$', '', aseg_data\$EID) # Ensure both patterns are removed
aseg_rds_file <- sub('csv$', 'rds', '$SUBJECTS_FOLDER/aseg_hbn.csv')
saveRDS(aseg_data, aseg_rds_file)

# Initialize list to store aparc data
aparc_data <- list()

# Process each measure
for (measure in c('volume', 'area', 'thickness')) {
    lh_file <- paste0('$SUBJECTS_FOLDER/aparc_lh_', measure, '_hbn.csv')
    rh_file <- paste0('$SUBJECTS_FOLDER/aparc_rh_', measure, '_hbn.csv')

    # Load left hemisphere data
    lh_data <- read.delim(lh_file, sep = '\t', header = TRUE, skip = 0)
    colnames(lh_data) <- make.names(colnames(lh_data))
    colnames(lh_data)[1] <- 'EID'
    lh_data\$EID <- gsub('^sub-|_acq-HCP$', '', lh_data\$EID)

    # Load right hemisphere data
    rh_data <- read.delim(rh_file, sep = '\t', header = TRUE, skip = 0)
    colnames(rh_data) <- make.names(colnames(rh_data))
    colnames(rh_data)[1] <- 'EID'
    rh_data\$EID <- gsub('^sub-|_acq-HCP$', '', rh_data\$EID)

    # Merge left and right hemisphere data
    combined_data <- merge(lh_data, rh_data, by = 'EID')

    # Remove columns containing 'eTIV' or 'BrainSegVolNotVent'
    columns_to_keep <- !grepl('eTIV|BrainSegVolNotVent', colnames(combined_data))
    combined_data <- combined_data[, columns_to_keep]

    # Store cleaned data in the list
    aparc_data[[measure]] <- combined_data
}

# Merge all measures into one data frame
merged_aparc <- Reduce(function(x, y) merge(x, y, by = 'EID'), aparc_data)

# Save the cleaned merged data to RDS
merged_rds_file <- paste0('$SUBJECTS_FOLDER/aparc_hbn.rds')
saveRDS(merged_aparc, merged_rds_file)

cat('TSVs successfully converted to RDS, unwanted features removed, and aparc files merged.\n')
"

# Step 7: Remove all CSV files
echo "Step 7: Removing all CSV files"
rm -f "$SUBJECTS_FOLDER"/*.csv

echo "All steps completed successfully!"
