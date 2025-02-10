# ----------------------------
# User-defined Parameters
# ----------------------------
id_col <- "idc"                    # Name of the identifier column
wave_col <- "wave"                 # Name of the wave column (new)
thickness_suffix <- "_thickavg"     # Suffix for thickness features
area_suffix <- "_surfarea"         # Suffix for area features
dataset_prefix <- "genr"           # Dataset prefix (e.g., "genr"); change as needed
exclusion_rate <- 0.0              # Exclusion threshold as a fraction of total features.
                                  # If set to 0.0, a fixed threshold (exclusion_threshold) will be used.
exclusion_threshold <- 10          # Fixed threshold (number of features) for exclusion when exclusion_rate == 0.0
remove_exact_duplicates <- TRUE    # If TRUE, remove rows that are exactly the same (by idc & wave)
handle_duplicates <- FALSE         # If TRUE, make duplicate (idc, wave) combinations unique and merge duplicate groups

# NEW parameters:
outlier_feature_threshold_multiplier <- 1.5  # Multiplier for IQR threshold to flag problematic features
exclude_outlier_features_from_count <- TRUE    # If TRUE, exclude features flagged as problematic from subject-level outlier counts

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

# Remove exact duplicate rows based on the combination of idc and wave
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

# Optionally, create a combined identifier for duplicate checking (if handle_duplicates is TRUE)
dat_thick$combined_id <- paste(dat_thick[[id_col]], dat_thick[[wave_col]], sep = "_")
if(handle_duplicates) {
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
}
dat_thick$combined_id <- NULL

# Number of thickness features (all columns except the identifier and wave)
num_features_thick <- ncol(dat_thick) - 2

# Calculate the lower and upper thresholds for each thickness feature (columns 3:ncol(dat_thick))
lower_thick <- rep(NA, num_features_thick)
upper_thick <- rep(NA, num_features_thick)
for(x in 3:ncol(dat_thick)) {
  this_col <- dat_thick[, x]
  lower_thick[x - 2] <- mean(this_col, na.rm = TRUE) - 2.698 * sd(this_col, na.rm = TRUE)
  upper_thick[x - 2] <- mean(this_col, na.rm = TRUE) + 2.698 * sd(this_col, na.rm = TRUE)
}

# For each subject (each row), count the number of thickness outliers (using all features initially)
thick_outlier_counts <- numeric(nrow(dat_thick))
for(i in 1:nrow(dat_thick)) {
  subj_values <- as.numeric(dat_thick[i, -(1:2)])
  lowind <- which(subj_values < lower_thick)
  upind  <- which(subj_values > upper_thick)
  thick_outlier_counts[i] <- length(lowind) + length(upind)
}

# Create a data frame with idc, wave, and thickness outlier count
df_thick <- data.frame(dat_thick[, c(id_col, wave_col), drop = FALSE],
                       thickness_outliers = thick_outlier_counts)

# --- NEW: Count outliers per thickness feature ---
# For each feature (columns 3:ncol(dat_thick)), count subjects with outliers.
thickness_feature_counts <- sapply(3:ncol(dat_thick), function(j) {
  sum(as.numeric(dat_thick[, j]) < lower_thick[j - 2] | as.numeric(dat_thick[, j]) > upper_thick[j - 2], na.rm = TRUE)
})
names(thickness_feature_counts) <- names(dat_thick)[3:ncol(dat_thick)]

cat("\nOutlier counts per thickness feature:\n")
print(thickness_feature_counts)

# --- NEW: Identify thickness features with significantly more outliers ---
thick_q1 <- quantile(thickness_feature_counts, 0.25)
thick_q3 <- quantile(thickness_feature_counts, 0.75)
thick_iqr <- thick_q3 - thick_q1
thick_threshold <- thick_q3 + outlier_feature_threshold_multiplier * thick_iqr

sig_thick_features <- thickness_feature_counts[thickness_feature_counts > thick_threshold]

cat("\nThickness features with significantly more outliers (IQR threshold =", thick_threshold, "):\n")
if(length(sig_thick_features) > 0) {
  print(sig_thick_features)
} else {
  cat("None found.\n")
}

# --- NEW: Optionally recalculate subject-level thickness outlier counts excluding problematic features ---
if (exclude_outlier_features_from_count) {
  thickness_feature_names <- names(dat_thick)[3:ncol(dat_thick)]
  non_prob_idx_thick <- which(!(thickness_feature_names %in% names(sig_thick_features)))

  thick_outlier_counts_new <- numeric(nrow(dat_thick))
  for(i in 1:nrow(dat_thick)) {
    subject_vals <- as.numeric(dat_thick[i, -(1:2)])
    selected_vals <- subject_vals[non_prob_idx_thick]
    selected_lower <- lower_thick[non_prob_idx_thick]
    selected_upper <- upper_thick[non_prob_idx_thick]
    low_count <- sum(selected_vals < selected_lower, na.rm = TRUE)
    up_count <- sum(selected_vals > selected_upper, na.rm = TRUE)
    thick_outlier_counts_new[i] <- low_count + up_count
  }
  df_thick$thickness_outliers <- thick_outlier_counts_new
  num_features_thick <- length(non_prob_idx_thick)
  cat("\nRecalculated thickness outlier counts excluding problematic features.\n")
}

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

# Remove exact duplicate rows based on the combination of idc and wave
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

# Optionally, create a combined identifier for area data
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

# Calculate the lower and upper thresholds for each area feature (columns 3:ncol(dat_area))
lower_area <- rep(NA, num_features_area)
upper_area <- rep(NA, num_features_area)
for(x in 3:ncol(dat_area)) {
  this_col <- dat_area[, x]
  lower_area[x - 2] <- mean(this_col, na.rm = TRUE) - 2.698 * sd(this_col, na.rm = TRUE)
  upper_area[x - 2] <- mean(this_col, na.rm = TRUE) + 2.698 * sd(this_col, na.rm = TRUE)
}

# For each subject, count the number of area outliers (using all features initially)
area_outlier_counts <- numeric(nrow(dat_area))
for(i in 1:nrow(dat_area)) {
  subj_values <- as.numeric(dat_area[i, -(1:2)])
  lowind <- which(subj_values < lower_area)
  upind  <- which(subj_values > upper_area)
  area_outlier_counts[i] <- length(lowind) + length(upind)
}

# Create a data frame with idc, wave, and area outlier count
df_area <- data.frame(dat_area[, c(id_col, wave_col), drop = FALSE],
                      area_outliers = area_outlier_counts)

# --- NEW: Count outliers per area feature ---
area_feature_counts <- sapply(3:ncol(dat_area), function(j) {
  sum(as.numeric(dat_area[, j]) < lower_area[j - 2] | as.numeric(dat_area[, j]) > upper_area[j - 2], na.rm = TRUE)
})
names(area_feature_counts) <- names(dat_area)[3:ncol(dat_area)]

cat("\nOutlier counts per area feature:\n")
print(area_feature_counts)

# --- NEW: Identify area features with significantly more outliers ---
area_q1 <- quantile(area_feature_counts, 0.25)
area_q3 <- quantile(area_feature_counts, 0.75)
area_iqr <- area_q3 - area_q1
area_threshold <- area_q3 + outlier_feature_threshold_multiplier * area_iqr

sig_area_features <- area_feature_counts[area_feature_counts > area_threshold]

cat("\nArea features with significantly more outliers (IQR threshold =", area_threshold, "):\n")
if(length(sig_area_features) > 0) {
  print(sig_area_features)
} else {
  cat("None found.\n")
}

# --- NEW: Optionally recalculate subject-level area outlier counts excluding problematic features ---
if (exclude_outlier_features_from_count) {
  area_feature_names <- names(dat_area)[3:ncol(dat_area)]
  non_prob_idx_area <- which(!(area_feature_names %in% names(sig_area_features)))

  area_outlier_counts_new <- numeric(nrow(dat_area))
  for(i in 1:nrow(dat_area)) {
    subject_vals <- as.numeric(dat_area[i, -(1:2)])
    selected_vals <- subject_vals[non_prob_idx_area]
    selected_lower <- lower_area[non_prob_idx_area]
    selected_upper <- upper_area[non_prob_idx_area]
    low_count <- sum(selected_vals < selected_lower, na.rm = TRUE)
    up_count <- sum(selected_vals > selected_upper, na.rm = TRUE)
    area_outlier_counts_new[i] <- low_count + up_count
  }
  df_area$area_outliers <- area_outlier_counts_new
  num_features_area <- length(non_prob_idx_area)
  cat("\nRecalculated area outlier counts excluding problematic features.\n")
}

# ----------------------------
# STEP 3: Merge Results and Create Summary
# ----------------------------
df_out <- merge(df_thick, df_area, by = c(id_col, wave_col), all = TRUE)
df_out[is.na(df_out)] <- 0
df_out$total_outliers <- df_out$thickness_outliers + df_out$area_outliers
total_features <- num_features_thick + num_features_area

# Create the exclusion flag:
# If exclusion_rate is 0.0, use the fixed threshold (exclusion_threshold); otherwise use the fraction.
if(exclusion_rate == 0.0) {
  df_out$exclude <- ifelse(df_out$total_outliers > exclusion_threshold, 1, 0)
} else {
  df_out$exclude <- ifelse(df_out$total_outliers > exclusion_rate * total_features, 1, 0)
}

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
  df_out$combined_id <- paste(df_out[[id_col]], df_out[[wave_col]], sep = "_")
  df_out$base_ID <- sub("_[0-9]+$", "", df_out$combined_id)
  dup_groups <- split(df_out, df_out$base_ID)
  merged_rows <- lapply(dup_groups, function(group) {
    if(nrow(group) > 1 && length(unique(group$total_outliers)) == 1) {
      group[1, id_col] <- strsplit(group$base_ID[1], "_")[[1]][1]
      group[1, wave_col] <- paste(strsplit(group$base_ID[1], "_")[[1]][-1], collapse = "_")
      return(group[1,])
    } else {
      return(group)
    }
  })
  df_out <- do.call(rbind, merged_rows)
  df_out$base_ID <- NULL
  df_out$combined_id <- NULL
} else {
  cat("Skipping merging of duplicate groups.\n")
}

# ----------------------------
# STEP 4: Add duplicate_removed Column and Save Summary to RDS
# ----------------------------
if(remove_exact_duplicates) {
  dup_ids_thick <- unique(orig_dat_thick[[id_col]][ duplicated(orig_dat_thick[, c(id_col, wave_col)]) |
                                                     duplicated(orig_dat_thick[, c(id_col, wave_col)], fromLast = TRUE) ])
  dup_ids_area <- unique(orig_dat_area[[id_col]][ duplicated(orig_dat_area[, c(id_col, wave_col)]) |
                                                   duplicated(orig_dat_area[, c(id_col, wave_col)], fromLast = TRUE) ])
  dup_ids_all <- unique(c(dup_ids_thick, dup_ids_area))
  df_out$duplicate_removed <- ifelse(sub("_[0-9]+$", "", df_out[[id_col]]) %in% dup_ids_all, 1, 0)
} else {
  df_out$duplicate_removed <- 0
}

cat("\nNumber of rows with duplicate_removed set to 1:", sum(df_out$duplicate_removed == 1), "\n")
saveRDS(df_out, file = paste0(dataset_prefix, "_qc.rds"))

# ----------------------------
# STEP 5: Additional Summary Reporting
# ----------------------------
cat("\nOccurrences of total_outliers values:\n")
print(table(df_out$total_outliers))

if(handle_duplicates) {
  df_out$combined_id <- paste(df_out[[id_col]], df_out[[wave_col]], sep = "_")
  df_out$base_ID <- sub("_[0-9]+$", "", df_out$combined_id)
  cat("\nDuplicate groups (by", id_col, "and", wave_col, ") with differing total_outliers values:\n")
  found_duplicate_diff <- FALSE
  dup_groups <- split(df_out, df_out$base_ID)
  for (group in dup_groups) {
    if(nrow(group) > 1) {
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
