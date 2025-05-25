#!/bin/bash

# Ensure the script stops on first error
set -e

# This script is designed to synchronize local and remote directories, excluding certain files and folders

############################## START EDITABLE AREA ##############################

# Source password script as passwordless login is currently not enabled
# The script should define `password_icb` variable
source "/Users/kemalinecik/Documents/Helmholtz/password.sh"

# Define constants
LOCAL_PATH="/Users/kemalinecik/git_nosync"
SERVER_PATH="/home/icb/kemal.inecik/work/codes"
PROJECT_NAME="idtrack"
REMOTE_USER="kemal.inecik"
REMOTE_HOST="hpc-build01"
REMOTE_PATH="$SERVER_PATH/$PROJECT_NAME"

EXCLUSIONS=("__pycache__" "*.DS_Store" ".idea" ".mypy_cache" "*.ipynb_checkpoints" ".git" "*_build")
SYNC_DIRS=("docs/_notebooks")

# Define ssh and rsync options
SSH_OPTIONS="ssh -t -o LogLevel=error"
RSYNC_OPTIONS="-avh --delete"

# Auto-sync Sleep
SLEEP_TIME=9

############################## END EDITABLE AREA ##############################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Function to print log messages with timestamp
source "$SCRIPT_DIR/utils/_log.sh"
source "$SCRIPT_DIR/utils/_colors.sh"

# Convert each exclusion into '--exclude=pattern' and save it to a new array
RSYNC_EXCLUSIONS=()
for exclusion in "${EXCLUSIONS[@]}"; do
    RSYNC_EXCLUSIONS+=("--exclude=$exclusion")
done

# Do the same for directories to be excluded
RSYNC_EXCLUSIONS_DIRS=("${RSYNC_EXCLUSIONS[@]}")
for dir in "${SYNC_DIRS[@]}"; do
    RSYNC_EXCLUSIONS_DIRS+=("--exclude=$dir")
done

print_rsync_output() {
    local rsync_output=$1
    if [[ $rsync_output == *">"* || $rsync_output == *"<"* || $rsync_output == *"deleting"* ]]; then
        echo "$rsync_output" | awk -v yellow="$YELLOW" -v red="$RED" -v reset="$RESET" '{
            if ($0 ~ /^>f.*.... /) {sub("^>f.*.... ", "> "); print yellow $0 reset}
            else if ($0 ~ /^<f.*.... /) {sub("^<f.*.... ", "< "); print yellow $0 reset}
            else if ($0 ~ /^\*deleting /) {sub("^\*deleting ", " "); print red $0 reset}
        }'
    else
        log "No updates were identified for synchronization."
    fi
}

# Define function for syncing directories
sync_dir() {
    local dir=$1
    log "Initiating synchronization for directory: '$dir'"

    # Get the parent directory of the target directory
    local parent_dir=$(dirname "$dir")

    # Check if local directory structure exists
    if [ ! -d "$LOCAL_PATH/$PROJECT_NAME/$parent_dir" ]; then
        log "Local directory '$LOCAL_PATH/$PROJECT_NAME/$parent_dir' does not exist. Exiting."
        exit 1
    fi

    local rsync_output=$(sshpass -p $password_icb rsync -e "$SSH_OPTIONS" $RSYNC_OPTIONS \
        "${RSYNC_EXCLUSIONS[@]}" \
        --itemize-changes \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$dir" \
        "$LOCAL_PATH/$PROJECT_NAME/$parent_dir") || {
            log "Error during rsync of '$dir'. Exiting."
            exit 1
        }
    print_rsync_output "$rsync_output"
    log "Synchronization completed for directory: '$dir'"
}

# Send local codes to remote excluding certain directories
send_local_codes() {
    log 'Initiating code transmission to remote server'
    local rsync_output=$(sshpass -p $password_icb rsync -e "$SSH_OPTIONS" $RSYNC_OPTIONS \
        "${RSYNC_EXCLUSIONS_DIRS[@]}" \
        --itemize-changes \
        "$LOCAL_PATH/$PROJECT_NAME" \
        "$REMOTE_USER@$REMOTE_HOST:$SERVER_PATH") || {
            log "Error during sending codes. Exiting."
            exit 1
        }
    print_rsync_output "$rsync_output"
    log 'Code transmission to remote server completed successfully'
}

# Function to send all codes excluding SYNC_DIRS but respecting EXCLUSIONS
send_all_ignore_sync_dirs() {
    log "Sending all files and folders to the server."
    local rsync_output=$(sshpass -p $password_icb rsync -e "$SSH_OPTIONS" $RSYNC_OPTIONS \
        "${RSYNC_EXCLUSIONS[@]}" \
        --itemize-changes \
        "$LOCAL_PATH/$PROJECT_NAME" \
        "$REMOTE_USER@$REMOTE_HOST:$SERVER_PATH") || {
            log "Error during sending codes. Exiting."
            exit 1
        }
    print_rsync_output "$rsync_output"
    log 'Transmission completed successfully'
}

# Function to get all codes from the server, excluding SYNC_DIRS but respecting EXCLUSIONS
get_all_from_the_server() {
    log "Retrieving all files and folders from the server."
    local rsync_output=$(sshpass -p $password_icb rsync -e "$SSH_OPTIONS" $RSYNC_OPTIONS \
        "${RSYNC_EXCLUSIONS[@]}" \
        --itemize-changes \
        "$REMOTE_USER@$REMOTE_HOST:$SERVER_PATH/$PROJECT_NAME" \
        "$LOCAL_PATH") || {
            log "Error during retrieving codes. Exiting."
            exit 1
        }
    print_rsync_output "$rsync_output"
    log 'Retrieval completed successfully'
}

# Function to confirm an operation
confirm_operation() {
    local operation_desc=$1
    local prompt_msg="$(date +'%Y-%m-%d %H:%M:%S') - Are you sure? $operation_desc"
    prompt_msg+=" This cannot be undone. (type 'yes' to confirm): "
    read -r -p "$prompt_msg" confirm
    if [[ $confirm == 'yes' ]]; then
        return 0
    else
        log "Operation cancelled."
        return 1
    fi
}

# Define function to check if files have changed
check_file_changes() {
    # Set the local directory to be observed
    local dir="$LOCAL_PATH/$PROJECT_NAME"
    # Initialize an array to hold the current md5 hashes
    local current_md5=()

    # Store the current Internal Field Separator (IFS)
    local oldIFS="$IFS"
    # Temporarily change IFS to newline character to handle filenames with spaces
    IFS=$'\n'

    # Construct the find command, excluding directories in RSYNC_EXCLUSIONS_DIRS
    local find_command="find $dir -type f"
    for excluded_dir in "${RSYNC_EXCLUSIONS_DIRS[@]}"; do
        # Remove the '--exclude=' part to get the actual directory name
        excluded_dir="${excluded_dir/--exclude=/}"
        find_command+=" ! -path \"*$excluded_dir*\""
    done

    # Execute the find command and store the resulting files in an array
    local files=()
    while IFS=  read -r -d $'\0'; do
        files+=("$REPLY")
    done < <(eval "$find_command -print0")

    # Reset IFS to its original value
    IFS="$oldIFS"

    # For each file found, compute the md5 hash
    for file in "${files[@]}"; do
        # Verify if the file still exists before computing the hash
        if [[ -e "$file" ]]; then
            current_md5+=("$(md5 -q "$file")")
        else
            echo "Error: File not found: $file"
        fi
    done

    # Check if the hash has changed since the last check
    if [[ $prev_md5 != "${current_md5[*]}" ]]; then
        prev_md5="${current_md5[*]}"
        send_local_codes
    else
        log "No updates were identified for synchronization. Press any key to stop..."
    fi
}

# Function for checking file changes
check_file_changes_loop() {
    while [ ! -s "$tmp" ]; do
        check_file_changes
        sleep "$SLEEP_TIME"
    done
}

# Main loop
trap '[[ -n $loop_pid ]] && kill -0 $loop_pid 2>/dev/null && kill $loop_pid' EXIT

# Function to handle SIGTERM signal
function handle_sigterm() {
    log "Received termination signal. Exiting..."
    exit 0
}

while true; do
    echo ''
    log "Please select an option for IDTrack package sync:"
    echo "1 - Send local codes"
    echo "2 - Send local codes and synchronize selected directories"
    echo "3 - Continuously monitor files for changes, synchronize if necessary"
    echo "4 - Send everything to the server"
    echo "5 - Get everything from the server"
    echo "q - Quit"

    # Prompt the user for input
    read -r -p "$(date +'%Y-%m-%d %H:%M:%S') - Enter [1/2/3/4/5/q]: " option
    case $option in
        1)
            send_local_codes
            ;;
        2)
            for dir in "${SYNC_DIRS[@]}"; do
                sync_dir "$dir"
            done
            send_local_codes
            ;;
        3)
            log "Monitoring files for changes. Press any key to stop..."
            # Start the checking loop in the background and trap SIGTERM signal
            trap 'handle_sigterm' TERM
            check_file_changes_loop &
            # Store the PID of the loop process
            loop_pid=$!
            # Wait for user input, then kill the loop process
            read -n 1
            kill $loop_pid
            log "Monitoring stopped."
            ;;
        4)
            if confirm_operation "You are about to send everything to the server."; then
                send_all_ignore_sync_dirs
            fi
            ;;
        5)
            if confirm_operation "You are about to retrieve everything from the server."; then
                get_all_from_the_server
            fi
            ;;
        q)
            log "Exiting script."
            break
            ;;
        *)
            log "Invalid option. Please enter 1, 2, 3, 4, 5 or 'q'."
            ;;
    esac
done

log 'Script execution completed successfully'
