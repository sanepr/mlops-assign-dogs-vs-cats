#!/usr/bin/env python3
"""
Script to rename all numeric-prefixed jpg files in data folders.
Adds cat_ prefix to files in cats folders and dog_ prefix to files in dogs folders.
"""
import os
import re

BASE_DIR = "/Users/aashishr/codebase/mlso_ass/data"

# Define all folders to process with their prefix
FOLDERS = [
    ("processed/train/cats", "cat_"),
    ("processed/val/cats", "cat_"),
    ("processed/test/cats", "cat_"),
    ("raw/cats", "cat_"),
    ("processed/train/dogs", "dog_"),
    ("processed/val/dogs", "dog_"),
    ("processed/test/dogs", "dog_"),
    ("raw/dogs", "dog_"),
]

def rename_files_in_folder(folder_path, prefix):
    """Rename all files starting with a digit to have the given prefix."""
    if not os.path.exists(folder_path):
        return 0, f"Folder not found: {folder_path}"

    count = 0
    for filename in os.listdir(folder_path):
        # Match files that start with a digit and end with .jpg
        if re.match(r'^[0-9].*\.jpg$', filename, re.IGNORECASE):
            old_path = os.path.join(folder_path, filename)
            new_filename = prefix + filename
            new_path = os.path.join(folder_path, new_filename)

            # Only rename if target doesn't exist
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                count += 1

    return count, f"Renamed {count} files in {folder_path}"

if __name__ == "__main__":
    total = 0
    results = []

    for rel_folder, prefix in FOLDERS:
        folder_path = os.path.join(BASE_DIR, rel_folder)
        count, message = rename_files_in_folder(folder_path, prefix)
        total += count
        results.append(message)

    # Write results to file
    output_file = "/Users/aashishr/codebase/mlso_ass/rename_results.txt"
    with open(output_file, "w") as f:
        for r in results:
            f.write(r + "\n")
        f.write(f"\nTotal files renamed: {total}\n")

    # Also print to stdout
    for r in results:
        print(r)
    print(f"\nTotal files renamed: {total}")
