#!/bin/bash

# Define user, host, and target directory
USER="rduijsens"
HOST="snellius.surf.nl"
SERVER="$USER@$HOST"
TARGET_DIR="mtproject"

# Define the local directories and files to sync
LOCAL_PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$0")")")"
SRC_DIR="$LOCAL_PROJECT_ROOT/src"
DATA_DIR="$LOCAL_PROJECT_ROOT/data"
CONFIG_DIR="$LOCAL_PROJECT_ROOT/config"
RUN_TRAINING_SCRIPT="$LOCAL_PROJECT_ROOT/scripts/slurm/run_training.sh"
RUN_TRAINING_GPU_SCRIPT="$LOCAL_PROJECT_ROOT/scripts/slurm/run_training_gpu.sh"
RUN_VALIDATION_SCRIPT="$LOCAL_PROJECT_ROOT/scripts/slurm/run_validation.sh"
RUN_VALIDATION_GPU_SCRIPT="$LOCAL_PROJECT_ROOT/scripts/slurm/run_validation_gpu.sh"
POETRY_LOCK="$LOCAL_PROJECT_ROOT/poetry.lock"
PYPROJECT_TOML="$LOCAL_PROJECT_ROOT/pyproject.toml"

# rsync options
RSYNC_OPTIONS=(-av --exclude="*.pyc" --exclude="__pycache__")

# Use rsync to copy directories and files with exclusions
rsync "${RSYNC_OPTIONS[@]}" "$SRC_DIR" "$SERVER:$TARGET_DIR"
#rsync "${RSYNC_OPTIONS[@]}" "$DATA_DIR" "$SERVER:$TARGET_DIR"
rsync "${RSYNC_OPTIONS[@]}" "$CONFIG_DIR" "$SERVER:$TARGET_DIR"
rsync "${RSYNC_OPTIONS[@]}" "$RUN_TRAINING_SCRIPT" "$SERVER:$TARGET_DIR/run_training.sh"
rsync "${RSYNC_OPTIONS[@]}" "$RUN_TRAINING_GPU_SCRIPT" "$SERVER:$TARGET_DIR/run_training_gpu.sh"
rsync "${RSYNC_OPTIONS[@]}" "$RUN_VALIDATION_SCRIPT" "$SERVER:$TARGET_DIR/run_validation.sh"
rsync "${RSYNC_OPTIONS[@]}" "$RUN_VALIDATION_GPU_SCRIPT" "$SERVER:$TARGET_DIR/run_validation_gpu.sh"
rsync "${RSYNC_OPTIONS[@]}" "$POETRY_LOCK" "$SERVER:$TARGET_DIR/poetry.lock"
rsync "${RSYNC_OPTIONS[@]}" "$PYPROJECT_TOML" "$SERVER:$TARGET_DIR/pyproject.toml"

echo "Files and directories successfully synced to $SERVER:$TARGET_DIR, excluding .pyc files and __pycache__ directories."
