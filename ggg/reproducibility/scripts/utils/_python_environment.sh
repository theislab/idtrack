#!/bin/bash

# Ensure the script stops on first error
set -e

# Source profile
source "$HOME/.bash_profile"
source "$(dirname "${BASH_SOURCE[0]}")/_log.sh"

# Check if the environment variable is a valid conda environment
if [[ $(conda env list | awk '{print $1}' | grep -Fx "$1") == "" ]]; then
    log "Error: $1 is not a valid conda environment."
    exit 1
fi

# Activate conda environment
log "Activating conda environment: $1..."
conda activate "$1"
if [ $? -ne 0 ]; then
    log "ERROR: Could not activate conda environment: $1"
    exit 1
fi

# Display the path of the current Python3 binary
log "Current Python3 path: $(which python3)"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    log "ERROR: Python3 could not be found"
    exit 1
fi
