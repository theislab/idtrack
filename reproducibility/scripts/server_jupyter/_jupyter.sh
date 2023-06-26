#!/bin/bash

# Ensure the script stops on first error
set -e

# Check the number of arguments
if [ "$#" -ne 2 ]; then
    echo "ERROR: This script expects two arguments: the conda environment and the path to the Python script."
    exit 1
fi

# Import and run the script
source "$2/../utils/_log.sh" "$1"
source "$2/../utils/_python_environment.sh" "$1"

# Run Python script
log "Running Python script: $2/utils/system.py..."
python3 "$2/../utils/_system.py"
if [ $? -ne 0 ]; then
    log "ERROR: Python script failed to run."
    exit 1
fi

# Start Jupyter Lab in $HOME directory
log "Starting Jupyter Lab at $HOME..."
(
cd "$HOME" || exit
jupyter lab --no-browser --ip 0.0.0.0
)

if [ $? -ne 0 ]; then
    log "ERROR: Jupyter Lab failed to start."
    exit 1
fi

log "Script executed successfully."
