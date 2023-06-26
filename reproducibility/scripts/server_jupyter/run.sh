#!/bin/bash

# Ensure the script stops on first error
set -e

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the output directory
OUTPUT_DIR="$HOME/.slurm_output_jupyter/%j.out"

# Define the username
USERNAME="kemal.inecik"

# slurm_output_dir
SLURM_OUTPUT_DIR="$HOME/.slurm_output_jupyter"

# Define option mappings
declare -A OPTION_MAPPING
OPTION_MAPPING=( ["cpu"]="cpu_p" ["icpu"]="interactive_cpu_p" ["gpu"]="gpu_p" ["igpu"]="interactive_gpu_p")

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <option> <env_variable>"
    echo "Options: cpu, icpu, gpu, igpu"
    exit 1
fi

# Function to print log messages with timestamp
source "$SCRIPT_DIR/../utils/_log.sh"

# Assign arguments to variables
option="$1"
env_variable="$2"

# Check if env_variable is a non-empty string
if [ -z "$env_variable" ]; then
    echo "Error: Environment variable must be a non-empty string."
    exit 1
fi

# Check if the option is one of the allowed options
case $option in
    cpu|icpu|gpu|igpu)
        ;;
    *)
        echo "Error: Option must be one of 'cpu', 'icpu', 'gpu', 'igpu'."
        exit 1
        ;;
esac

# Common code to be executed for all options
log "Script started."
source "$HOME/.bash_profile"
source "$SCRIPT_DIR/_slurm_output_dir.sh" "$SCRIPT_DIR" "$SLURM_OUTPUT_DIR"

# Create the export section variable
export_section="exported=\"$env_variable|$SCRIPT_DIR\""

# Run the code based on the provided option
log "Running $option section..."
sbatch --export=$export_section -o "$OUTPUT_DIR" "$SCRIPT_DIR/_slurm_job_${option}.sbatch"
sleep 2
source "$SCRIPT_DIR/_squeue.sh" "$USERNAME" "${OPTION_MAPPING[$option]}" "$SCRIPT_DIR"

log "Script ended."
