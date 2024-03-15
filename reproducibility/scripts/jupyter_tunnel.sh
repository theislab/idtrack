#!/bin/bash

# Ensure the script stops on first error
set -e

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Function to print log messages with timestamp
source "$SCRIPT_DIR/utils/_log.sh"
source "$SCRIPT_DIR/utils/_colors.sh"

# Ask the user for the URL
log "Please enter the URL:"
read url

# Regular expression pattern to match the URL format and capture the hostname
url_pattern="^http:\/\/([^.]+)\.(.+):([0-9]+)\/lab\?token=[a-f0-9]{48}$"

# Check if the provided URL matches the pattern and extract the hostname
if [[ $url =~ $url_pattern ]]; then
    hostname="${BASH_REMATCH[1]}"
    domain="${BASH_REMATCH[2]}"
    jupyter_port="${BASH_REMATCH[3]}"
else
    log "The provided URL does not match the specified format."
    exit 1
fi

# Source password script as passwordless login is currently not enabled
# The script should define `password_icb` variable
source "/Users/kemalinecik/Documents/Helmholtz/password.sh"

REMOTE_USER="kemal.inecik"
REMOTE_HOST="hpc-build01"
LOCAL_PORT="10101"

# Build the SSH command
sshpass_command="sshpass -p '$password_icb'"
ssh_command="ssh -t -o LogLevel=error -L $LOCAL_PORT:$hostname:$jupyter_port $REMOTE_USER@$REMOTE_HOST"

# Modify the URL
modified_url=$(echo "$url" | sed "s/$hostname.$domain:$jupyter_port/localhost:$LOCAL_PORT/g")

# Print the modified URL in a different color
echo -e "${YELLOW}$modified_url${RESET}"

# Print the SSH command in color
echo -e "${BLUE}SSH connection is building..${RESET}"
ssh_command_with_password="$sshpass_command $ssh_command"
echo -e "${BLUE}SSH Command: $ssh_command${RESET}"

# Execute the SSH command
eval "$ssh_command_with_password"

# Exit the script
exit 0
