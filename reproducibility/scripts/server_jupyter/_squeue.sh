#!/bin/bash

# Ensure the script stops on first error
set -e

# Function to print log messages with timestamp
source "$3/../utils/_log.sh"

# Check if both arguments are provided
if [ $# -ne 3 ]; then
    log "Usage: _squeue.sh <username> <partition> <parent_script_dir>"
    exit 1
fi

username=$1
partition=$2

# Run the squeue command with the provided arguments
squeue -u "$username" -p "$partition" -o "%.18i %.30j %.20u %.8T %.3C %.5m %.6y %.10M %.10l %.3D %R" -S "N"
