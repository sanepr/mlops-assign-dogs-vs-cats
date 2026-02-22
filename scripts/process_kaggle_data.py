#!/usr/bin/env python3
"""
Script to process Kaggle PetImages dataset and prepare it for training.
Handles both the Kaggle format (Cat/Dog folders) and custom format (cats/dogs).
"""
import os
import shutil
import random
from PIL import Image
import argparse

def is_valid_image(filepath):
    """Check if a file is a valid image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def process_kaggle_data(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Process Kaggle PetImages dataset.

    Args:
        source_dir: Path to PetImages folder (contains Cat and Dog subfolders)
        output_dir: Path to output processed data
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
    """
    print(f"Processing Kaggle data from: {source_dir}")
    print(f"Output directory: {output_dir}")

    # Handle different folder naming conventions
    cat_folder = None
    dog_folder = None

    for name in ['Cat', 'cat', 'cats', 'Cats']:
        path = os.path.join(source_dir, name)
        if os.path.exists(path):
            cat_folder = path
            break

    for name in ['Dog', 'dog', 'dogs', 'Dogs']:
        path = os.path.join(source_dir, name)
        if os.path.exists(path):
            dog_folder = path
            break

    if not cat_folder or not dog_folder:
        print(f"ERROR: Could not find Cat/Dog folders in {source_dir}")
        print(f"Contents: {os.listdir(source_dir)}")
        return False

    print(f"Found cat folder: {cat_folder}")
    print(f"Found dog folder: {dog_folder}")

    # Create output directories
    for split in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # Process each class
    for class_name, source_folder, target_name in [
        ('Cat', cat_folder, 'cats'),
        ('Dog', dog_folder, 'dogs')
    ]:
        print(f"\nProcessing {class_name} images...")

        # Get all valid images
        valid_images = []
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(source_folder, filename)
                if is_valid_image(filepath):
                    valid_images.append(filename)
                else:
                    print(f"  Skipping invalid image: {filename}")

        print(f"  Found {len(valid_images)} valid images")

        # Shuffle and split
        random.seed(42)
        random.shuffle(valid_images)

        n_total = len(valid_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_images = valid_images[:n_train]
        val_images = valid_images[n_train:n_train + n_val]
        test_images = valid_images[n_train + n_val:]

        print(f"  Split: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

        # Copy images to respective folders
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            dest_folder = os.path.join(output_dir, split_name, target_name)
            for i, filename in enumerate(split_images):
                src = os.path.join(source_folder, filename)
                # Rename to consistent format
                ext = os.path.splitext(filename)[1]
                new_name = f"{target_name[:-1]}_{i:04d}{ext}"
                dst = os.path.join(dest_folder, new_name)

                try:
                    # Copy and resize to 224x224
                    with Image.open(src) as img:
                        img = img.convert('RGB')
                        img = img.resize((224, 224), Image.LANCZOS)
                        img.save(dst, 'JPEG', quality=95)
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")

    print("\n✅ Data processing complete!")

    # Print summary
    for split in ['train', 'val', 'test']:
        cats = len(os.listdir(os.path.join(output_dir, split, 'cats')))
        dogs = len(os.listdir(os.path.join(output_dir, split, 'dogs')))
        print(f"  {split}: {cats} cats, {dogs} dogs")

    return True


def main():
    parser = argparse.ArgumentParser(description='Process Kaggle PetImages dataset')
    parser.add_argument('--source', type=str, default='data/raw/PetImages',
                        help='Path to PetImages folder')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training data ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation data ratio (default: 0.1)')

    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    success = process_kaggle_data(
        args.source,
        args.output,
        args.train_ratio,
        args.val_ratio,
        test_ratio
    )

    if success:
        print("\n🎉 Dataset is ready for training!")
        print(f"Run: python src/train.py --data-dir {args.output}/train --epochs 10")


if __name__ == '__main__':
    main()
