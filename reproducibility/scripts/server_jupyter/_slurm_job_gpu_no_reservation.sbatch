#!/bin/bash

#SBATCH -J bash_gpu
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH -t 23:59:00
#SBATCH -c 6
#SBATCH --mem=90G
#SBATCH --nice=0

export_section="${exported}"
env_variable=$(echo $export_section | awk -F '|' '{print $1}')
script_dir=$(echo $export_section | awk -F '|' '{print $2}')
SCRIPT_DIR=${script_dir//\"}
ENV_VAR=${env_variable//\"}

source "$SCRIPT_DIR/_jupyter.sh" "$ENV_VAR" "$SCRIPT_DIR"
