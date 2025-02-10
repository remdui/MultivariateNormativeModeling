# ----------------------------
# User-defined Parameters
# ----------------------------
id_col <- "idc"                    # Name of the identifier column
wave_col <- "wave"                 # Name of the wave column (new)
thickness_suffix <- "_thickavg"     # Suffix for thickness features
area_suffix <- "_surfarea"         # Suffix for area features
dataset_prefix <- "genr"           # Dataset prefix (e.g., "hbn"); change as needed
exclusion_rate <- 0.1              # Exclusion threshold as a fraction of total features
remove_exact_duplicates <- TRUE    # If TRUE, remove rows that are exactly the same (by idc & wave)
handle_duplicates <- FALSE         # If TRUE, make duplicate (idc, wave) combinations unique and merge duplicate groups

# ----------------------------
# Build file names based on the dataset prefix
# ----------------------------
thickness_file <- paste0(dataset_prefix, "_aparc_thickavg.rds")
area_file <- paste0(dataset_prefix, "_aparc_surfarea.rds")

# ----------------------------
# STEP 1: Process Thickness Data
# ----------------------------

dat_thick <- readRDS(thickness_file)

# Keep only columns that are the identifier, the wave variable, or end with the thickness suffix
keep_cols <- grep(paste0("^(", id_col, "|", wave_col, "$|.*", thickness_suffix, "$)"),
                  names(dat_thick), value = TRUE)
dat_thick <- dat_thick[, keep_cols]

# Save a copy of the original thickness data (after subsetting) for duplicate flagging
if(remove_exact_duplicates) {
  orig_dat_thick <- dat_thick
}

# Remove exact duplicate rows based on the combination of id_col and wave_col
if(remove_exact_duplicates) {
  n_thick_exact_duplicates <- sum(duplicated(dat_thick[, c(id_col, wave_col)]))
  if(n_thick_exact_duplicates > 0) {
    cat("Found", n_thick_exact_duplicates, "exact duplicate rows in", thickness_file,
        "based on", id_col, "and", wave_col, ". Removing them...\n")
    dat_thick <- dat_thick[!duplicated(dat_thick[, c(id_col, wave_col)]), ]
  } else {
    cat("No exact duplicate rows found in", thickness_file, "\n")
  }
}

# Ensure the identifier and wave columns are treated as character vectors
dat_thick[[id_col]] <- as.character(dat_thick[[id_col]])
dat_thick[[wave_col]] <- as.character(dat_thick[[wave_col]])

# Optionally, if you wish to check for duplicates based on the (id, wave) combination and make them unique,
# create a combined identifier.
dat_thick$combined_id <- paste(dat_thick[[id_col]], dat_thick[[wave_col]], sep = "_")

if(handle_duplicates) {
  # Check for duplicate (id, wave) combinations and, if found, append suffixes to make them unique
  n_duplicates_thick <- sum(duplicated(dat_thick$combined_id))
  if(n_duplicates_thick != 0) {
    cat("Found", n_duplicates_thick, "duplicate combinations of", id_col, "and", wave_col, "in", thickness_file,
        "file. Making them unique by appending suffixes...\n")
    dat_thick$combined_id <- ave(dat_thick$combined_id, dat_thick$combined_id, FUN = function(x) {
      if(length(x) == 1) x else paste0(x, "_", seq_along(x))
    })
  } else {
    cat("No duplicate combinations of", id_col, "and", wave_col, "values found in", thickness_file, "\n")
  }
  # (Optional) You can use dat_thick$combined_id as the unique key downstream.
}

dat_thick$combined_id <- NULL

# Number of thickness features (all columns except the identifier and wave)
num_features_thick <- ncol(dat_thick) - 2

# Calculate the lower and upper thresholds for each thickness feature
lower_thick <- rep(NA, num_features_thick)
upper_thick <- rep(NA, num_features_thick)
for(x in 3:ncol(dat_thick)){  # starting at column 3 because 1 = id, 2 = wave
  this_col <- dat_thick[, x]
  lower_thick[x - 2] <- mean(this_col, na.rm = TRUE) - 2.698 * sd(this_col, na.rm = TRUE)
  upper_thick[x - 2] <- mean(this_col, na.rm = TRUE) + 2.698 * sd(this_col, na.rm = TRUE)
}

# For each subject (i.e. each unique combination of id and wave), count the number of thickness outliers
thick_outlier_counts <- numeric(nrow(dat_thick))
for(i in 1:nrow(dat_thick)){
  # Check only the thickness feature columns (i.e., excluding id and wave)
  subj_values <- dat_thick[i, -(1:2)]
  lowind <- which(as.numeric(subj_values) < lower_thick)
  upind  <- which(as.numeric(subj_values) > upper_thick)
  thick_outlier_counts[i] <- length(lowind) + length(upind)
}

# Create a data frame with the identifier, wave, and thickness outlier count
df_thick <- data.frame(dat_thick[, c(id_col, wave_col), drop = FALSE],
                       thickness_outliers = thick_outlier_counts)

# ----------------------------
# STEP 2: Process Surface Area Data
# ----------------------------

dat_area <- readRDS(area_file)

# Keep only columns that are the identifier, the wave variable, or end with the area suffix
keep_cols <- grep(paste0("^(", id_col, "|", wave_col, "$|.*", area_suffix, "$)"),
                  names(dat_area), value = TRUE)
dat_area <- dat_area[, keep_cols]

# Save a copy of the original area data (after subsetting) for duplicate flagging
if(remove_exact_duplicates) {
  orig_dat_area <- dat_area
}

# Remove exact duplicate rows based on the combination of id_col and wave_col
if(remove_exact_duplicates) {
  n_area_exact_duplicates <- sum(duplicated(dat_area[, c(id_col, wave_col)]))
  if(n_area_exact_duplicates > 0) {
    cat("Found", n_area_exact_duplicates, "exact duplicate rows in", area_file,
        "based on", id_col, "and", wave_col, ". Removing them...\n")
    dat_area <- dat_area[!duplicated(dat_area[, c(id_col, wave_col)]), ]
  } else {
    cat("No exact duplicate rows found in", area_file, "\n")
  }
}

# Ensure the identifier and wave columns are treated as character vectors
dat_area[[id_col]] <- as.character(dat_area[[id_col]])
dat_area[[wave_col]] <- as.character(dat_area[[wave_col]])

# Optionally, create a combined identifier for area data as well
dat_area$combined_id <- paste(dat_area[[id_col]], dat_area[[wave_col]], sep = "_")

if(handle_duplicates) {
  n_duplicates_area <- sum(duplicated(dat_area$combined_id))
  if(n_duplicates_area != 0) {
    cat("Found", n_duplicates_area, "duplicate combinations of", id_col, "and", wave_col, "in", area_file,
        "file. Making them unique by appending suffixes...\n")
    dat_area$combined_id <- ave(dat_area$combined_id, dat_area$combined_id, FUN = function(x) {
      if(length(x) == 1) x else paste0(x, "_", seq_along(x))
    })
  } else {
    cat("No duplicate combinations of", id_col, "and", wave_col, "values found in", area_file, "\n")
  }
}

dat_area$combined_id <- NULL

# Number of area features (all columns except the identifier and wave)
num_features_area <- ncol(dat_area) - 2

# Calculate the lower and upper thresholds for each area feature
lower_area <- rep(NA, num_features_area)
upper_area <- rep(NA, num_features_area)
for(x in 3:ncol(dat_area)){  # starting at column 3 because 1 = id, 2 = wave
  this_col <- dat_area[, x]
  lower_area[x - 2] <- mean(this_col, na.rm = TRUE) - 2.698 * sd(this_col, na.rm = TRUE)
  upper_area[x - 2] <- mean(this_col, na.rm = TRUE) + 2.698 * sd(this_col, na.rm = TRUE)
}

# For each subject, count the number of area outliers
area_outlier_counts <- numeric(nrow(dat_area))
for(i in 1:nrow(dat_area)){
  subj_values <- dat_area[i, -(1:2)]
  lowind <- which(as.numeric(subj_values) < lower_area)
  upind  <- which(as.numeric(subj_values) > upper_area)
  area_outlier_counts[i] <- length(lowind) + length(upind)
}

# Create a data frame with the identifier, wave, and area outlier count
df_area <- data.frame(dat_area[, c(id_col, wave_col), drop = FALSE],
                      area_outliers = area_outlier_counts)

# ----------------------------
# STEP 3: Merge Results and Create Summary
# ----------------------------

# Merge the two data frames by the identifier and wave columns.
# (Assumes the same subjects are present in both files.)
df_out <- merge(df_thick, df_area, by = c(id_col, wave_col), all = TRUE)

# Replace any NA with 0 (if a subject is missing from one dataset)
df_out[is.na(df_out)] <- 0

# Calculate total outliers per subject
df_out$total_outliers <- df_out$thickness_outliers + df_out$area_outliers

# Total number of features assessed
total_features <- num_features_thick + num_features_area

# Create an "exclude" flag:
# If more than exclusion_rate of all features are outliers, then exclude = 1 (TRUE), else 0 (FALSE)
df_out$exclude <- ifelse(df_out$total_outliers > exclusion_rate * total_features, 1, 0)

# If duplicates are not being handled, check for duplicate (id, wave) combinations and exit gracefully if found.
combined_key <- paste(df_out[[id_col]], df_out[[wave_col]], sep = "_")
if(!handle_duplicates) {
  if(anyDuplicated(combined_key) != 0) {
    cat("Duplicate (", id_col, ",", wave_col, ") combinations found in the merged output and handle_duplicates is set to FALSE.\n", sep = "")
    stop("Exiting gracefully: Please resolve duplicates or set handle_duplicates = TRUE.")
  }
}

# ----------------------------
# STEP 3.5: Merge Duplicate Groups with Identical Outlier Counts (if handling duplicates)
# ----------------------------
if(handle_duplicates) {
  # Create a combined identifier in the merged data
  df_out$combined_id <- paste(df_out[[id_col]], df_out[[wave_col]], sep = "_")

  # Create a base_ID by stripping any appended suffix (e.g., "_2", "_3", etc.)
  df_out$base_ID <- sub("_[0-9]+$", "", df_out$combined_id)

  # Split the data by base_ID to group originally duplicate entries
  dup_groups <- split(df_out, df_out$base_ID)

  # For each group, if there is more than one row and all have the same total_outliers value,
  # merge them by keeping just one row (and setting the identifier and wave columns to the original values)
  merged_rows <- lapply(dup_groups, function(group) {
    if(nrow(group) > 1 && length(unique(group$total_outliers)) == 1) {
      # Optionally, you may reset the combined_id to the base value:
      group[1, id_col] <- strsplit(group$base_ID[1], "_")[[1]][1]
      group[1, wave_col] <- paste(strsplit(group$base_ID[1], "_")[[1]][-1], collapse = "_")
      return(group[1,])
    } else {
      return(group)
    }
  })
  df_out <- do.call(rbind, merged_rows)

  # (Optional) Remove the temporary base_ID and combined_id columns
  df_out$base_ID <- NULL
  df_out$combined_id <- NULL
} else {
  cat("Skipping merging of duplicate groups.\n")
}

# ----------------------------
# STEP 4: Add duplicate_removed Column and Save Summary to CSV
# ----------------------------

# Compute a duplicate flag based on the original datasets.
if(remove_exact_duplicates) {
  # Identify duplicate rows (by id and wave) in the thickness data:
  dup_ids_thick <- unique(orig_dat_thick[[id_col]][ duplicated(orig_dat_thick[, c(id_col, wave_col)]) |
                                                     duplicated(orig_dat_thick[, c(id_col, wave_col)], fromLast = TRUE) ])
  # Identify duplicate rows (by id and wave) in the area data:
  dup_ids_area <- unique(orig_dat_area[[id_col]][ duplicated(orig_dat_area[, c(id_col, wave_col)]) |
                                                   duplicated(orig_dat_area[, c(id_col, wave_col)], fromLast = TRUE) ])
  # Combine the IDs from both datasets.
  dup_ids_all <- unique(c(dup_ids_thick, dup_ids_area))

  # For the final merged output, check for duplicates by (id, wave) using the id_col values.
  # (If you later handled duplicates by appending suffixes, you may want to remove that suffix first.)
  df_out$duplicate_removed <- ifelse(sub("_[0-9]+$", "", df_out[[id_col]]) %in% dup_ids_all, 1, 0)
} else {
  df_out$duplicate_removed <- 0
}

cat("\nNumber of rows with duplicate_removed set to 1:", sum(df_out$duplicate_removed == 1), "\n")

# Save the final summary (including the duplicate_removed column) to an RDS file
saveRDS(df_out, file = paste0(dataset_prefix, "_qc.rds"))

# ----------------------------
# STEP 5: Additional Summary Reporting
# ----------------------------

cat("\nOccurrences of total_outliers values:\n")
print(table(df_out$total_outliers))

if(handle_duplicates) {
  # Create a combined identifier for duplicate-group checking
  df_out$combined_id <- paste(df_out[[id_col]], df_out[[wave_col]], sep = "_")
  df_out$base_ID <- sub("_[0-9]+$", "", df_out$combined_id)

  cat("\nDuplicate groups (by", id_col, "and", wave_col, ") with differing total_outliers values:\n")
  found_duplicate_diff <- FALSE
  dup_groups <- split(df_out, df_out$base_ID)
  for (group in dup_groups) {
    if(nrow(group) > 1) {  # More than one entry for this base combination
      if(length(unique(group$total_outliers)) > 1) {
        found_duplicate_diff <- TRUE
        cat("Group for", group$base_ID[1], ":\n")
        print(group)
      }
    }
  }
  if(!found_duplicate_diff) {
    cat("None found.\n")
  }
} else {
  cat("\nSkipping duplicate group reporting since duplicates were not handled.\n")
}

# Finally, print the number of flagged IDs and list them (flagged if exclude == 1)
flagged_ids <- paste(df_out[[id_col]], df_out[[wave_col]], sep = "_")[df_out$exclude == 1]
cat("\nNumber of flagged IDs:", length(flagged_ids), "\n")
if(length(flagged_ids) > 0) {
  cat("Flagged IDs:", paste(flagged_ids, collapse = ", "), "\n")
}

# ----------------------------
# STEP 6: Report Duplicate Row Removal Summary
# ----------------------------
if(remove_exact_duplicates) {
  cat("\nDuplicate Row Removal Summary:\n")
  cat("Thickness file (", thickness_file, "): Found and removed ", n_thick_exact_duplicates, " exact duplicate rows.\n", sep = "")
  cat("Area file (", area_file, "): Found and removed ", n_area_exact_duplicates, " exact duplicate rows.\n", sep = "")
}
