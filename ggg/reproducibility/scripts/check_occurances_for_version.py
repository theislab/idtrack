#!/usr/bin/env python3

import os


def find_string_in_files(directory: str, search_string: str) -> tuple:
    """Searches for a specific string in all files for given directory.

    Args:
        directory (str): Path of the search space
        search_string (str): String to be searched

    Returns:
        tuple: matching_files, total_files_checked, total_unreadable_files
    """
    matches = []
    files_checked = 0
    unreadable_files = 0
    idtrack_root_path = os.path.abspath(directory)

    for root, _, files in os.walk(idtrack_root_path, topdown=True):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, encoding="utf-8") as file:
                    content = file.readlines()
                    files_checked += 1
                    for line_number, line in enumerate(content, start=1):
                        if search_string in line:
                            matches.append((file_path, line_number))
            except (OSError, UnicodeDecodeError):
                unreadable_files += 1

    return matches, files_checked, unreadable_files


def main():
    """Main."""
    # Assuming this script is located at 'idtrack/reproducibility/version/check.py'
    # script_directory = os.path.dirname(os.path.realpath(__file__))
    idtrack_directory = "/Users/kemalinecik/git_nosync/idtrack"

    search_string = input("Enter the search string: ")

    matching_files, total_files_checked, total_unreadable_files = find_string_in_files(idtrack_directory, search_string)

    print(f"Total files checked: {total_files_checked}")
    print(f"Unreadable files: {total_unreadable_files}")
    if matching_files:
        print(f"Found the string {search_string!r} in the following files:")
        for file_path, line_number in matching_files:
            # Formatting path as a clickable link in VSCode
            print(f"{file_path}:{line_number}")
    else:
        print(f"No files found containing the string {search_string!r}.")


if __name__ == "__main__":
    main()
