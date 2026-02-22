#!/usr/bin/env python3
"""
Script to rename all numeric-prefixed jpg files in data folders.
Adds cat_ prefix to files in cats folders and dog_ prefix to files in dogs folders.
"""
import os
import re
import sys

BASE_DIR = "/Users/aashishr/codebase/mlso_ass/data"

# Folders to process
folders_to_process = [
    ("processed/train/cats", "cat_"),
    ("processed/val/cats", "cat_"),
    ("processed/test/cats", "cat_"),
    ("raw/cats", "cat_"),
    ("processed/train/dogs", "dog_"),
    ("processed/val/dogs", "dog_"),
    ("processed/test/dogs", "dog_"),
    ("raw/dogs", "dog_"),
]

def rename_files(folder_path, prefix):
    """Rename all numeric-prefixed jpg files with the given prefix."""
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return 0

    count = 0
    # Pattern to match files starting with a digit and ending with .jpg
    pattern = re.compile(r'^[0-9].*\.jpg$', re.IGNORECASE)

    for filename in os.listdir(folder_path):
        if pattern.match(filename):
            old_path = os.path.join(folder_path, filename)
            new_filename = f"{prefix}{filename}"
            new_path = os.path.join(folder_path, new_filename)

            # Only rename if destination doesn't exist
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                count += 1

    return count

def main():
    total_renamed = 0

    for folder, prefix in folders_to_process:
        full_path = os.path.join(BASE_DIR, folder)
        count = rename_files(full_path, prefix)
        print(f"Renamed {count} files in {folder}")
        total_renamed += count

    print(f"\nTotal files renamed: {total_renamed}")

if __name__ == "__main__":
    main()
