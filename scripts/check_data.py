#!/usr/bin/env python3
"""Check data status and move PetImages if needed."""
import os
import shutil

PROJECT_DIR = '/Users/aashishr/codebase/mlso_ass'
DOWNLOADS_DIR = '/Users/aashishr/Downloads'

def check_and_move_data():
    print("=== Checking Data Status ===\n")

    # Check Downloads for PetImages
    pet_images_download = os.path.join(DOWNLOADS_DIR, 'PetImages')
    if os.path.exists(pet_images_download):
        print(f"✅ Found PetImages in Downloads")

        # Move to project
        dest = os.path.join(PROJECT_DIR, 'data', 'raw', 'PetImages')
        if not os.path.exists(dest):
            shutil.move(pet_images_download, dest)
            print(f"   Moved to: {dest}")
        else:
            print(f"   Already exists at: {dest}")
    else:
        print(f"❌ PetImages not found in Downloads")

    # Check data/raw
    raw_dir = os.path.join(PROJECT_DIR, 'data', 'raw')
    print(f"\n📁 Contents of data/raw/:")
    for item in os.listdir(raw_dir):
        item_path = os.path.join(raw_dir, item)
        if os.path.isdir(item_path):
            count = len([f for f in os.listdir(item_path) if not f.startswith('.')])
            print(f"   {item}/: {count} files")
        else:
            print(f"   {item}")

    # Check for cats/dogs folders
    cats_dir = os.path.join(raw_dir, 'cats')
    dogs_dir = os.path.join(raw_dir, 'dogs')

    cats_count = 0
    dogs_count = 0

    if os.path.exists(cats_dir):
        cats_count = len([f for f in os.listdir(cats_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if os.path.exists(dogs_dir):
        dogs_count = len([f for f in os.listdir(dogs_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    print(f"\n📊 Image Counts:")
    print(f"   Cats: {cats_count}")
    print(f"   Dogs: {dogs_count}")
    print(f"   Total: {cats_count + dogs_count}")

    if cats_count > 1000:
        print("\n✅ This appears to be the full Kaggle dataset!")
    elif cats_count > 0:
        print("\n⚠️ This appears to be sample data. For better model accuracy,")
        print("   download the full dataset from Kaggle.")

    # Check for PetImages subfolder (Kaggle format)
    pet_images_in_raw = os.path.join(raw_dir, 'PetImages')
    if os.path.exists(pet_images_in_raw):
        print(f"\n📁 Found PetImages folder. Contents:")
        for item in os.listdir(pet_images_in_raw):
            item_path = os.path.join(pet_images_in_raw, item)
            if os.path.isdir(item_path):
                count = len([f for f in os.listdir(item_path) if not f.startswith('.')])
                print(f"   {item}/: {count} files")

        print("\n💡 Run the following to process this data:")
        print(f"   python scripts/process_kaggle_data.py --source data/raw/PetImages")

if __name__ == '__main__':
    check_and_move_data()
