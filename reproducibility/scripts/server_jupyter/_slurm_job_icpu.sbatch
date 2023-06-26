#!/bin/bash

#SBATCH -J bash_icpu
#SBATCH -p interactive_cpu_p
#SBATCH --qos=interactive_cpu
#SBATCH -t 11:59:00
#SBATCH -c 8
#SBATCH --mem=300G
#SBATCH --nice=0

export_section="${exported}"
env_variable=$(echo $export_section | awk -F '|' '{print $1}')
script_dir=$(echo $export_section | awk -F '|' '{print $2}')
SCRIPT_DIR=${script_dir//\"}
ENV_VAR=${env_variable//\"}

source "$SCRIPT_DIR/_jupyter.sh" "$ENV_VAR" "$SCRIPT_DIR"
