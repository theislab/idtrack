#!/bin/bash

# Ensure the script stops on first error
set -e

# Function to print log messages with timestamp
source "$1/../utils/_log.sh"

if [ ! -d "$2" ]; then
    mkdir "$2"
    log "Directory created: $2"
else
    log "Directory already exists: $2"
fi
