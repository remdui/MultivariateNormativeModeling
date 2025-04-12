#!/bin/bash

# Define user, host, and target directory
USER="rduijsens"
HOST="snellius.surf.nl"
SERVER="$USER@$HOST"
TARGET_DIR="mtproject"

# Define the local directories and files to sync
LOCAL_PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$0")")")"
SRC_DIR="$LOCAL_PROJECT_ROOT/src"
#CONFIG_DIR="$LOCAL_PROJECT_ROOT/config"
POETRY_LOCK="$LOCAL_PROJECT_ROOT/poetry.lock"
PYPROJECT_TOML="$LOCAL_PROJECT_ROOT/pyproject.toml"

# rsync options
RSYNC_OPTIONS=(-av --exclude="*.pyc" --exclude="__pycache__")

# Use rsync to copy directories and files with exclusions
rsync "${RSYNC_OPTIONS[@]}" "$SRC_DIR" "$SERVER:$TARGET_DIR"
#rsync "${RSYNC_OPTIONS[@]}" "$CONFIG_DIR" "$SERVER:$TARGET_DIR"
rsync "${RSYNC_OPTIONS[@]}" "$POETRY_LOCK" "$SERVER:$TARGET_DIR/poetry.lock"
rsync "${RSYNC_OPTIONS[@]}" "$PYPROJECT_TOML" "$SERVER:$TARGET_DIR/pyproject.toml"

echo "Files and directories successfully synced to $SERVER:$TARGET_DIR, excluding .pyc files and __pycache__ directories."
