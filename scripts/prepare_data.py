"""
Data Preparation Script
Downloads and prepares the Cats vs Dogs dataset
"""
import os
import sys
import argparse
import shutil
import random
import zipfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_dataset(output_dir: str):
    """
    Download the Cats vs Dogs dataset from Kaggle.

    Note: Requires kaggle API credentials to be set up.
    Alternatively, you can manually download from:
    https://www.kaggle.com/datasets/salader/dogs-vs-cats
    """
    try:
        import kaggle

        print("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            'salader/dogs-vs-cats',
            path=output_dir,
            unzip=True
        )
        print(f"Dataset downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Please download manually from Kaggle and place in data/raw/")
        return False


def create_sample_dataset(output_dir: str, num_samples: int = 100):
    """
    Create a sample dataset with placeholder images for testing.
    """
    from PIL import Image
    import numpy as np

    print(f"Creating sample dataset with {num_samples} images per class...")

    for cls in ['cats', 'dogs']:
        class_dir = os.path.join(output_dir, cls)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(num_samples):
            # Create random image
            if cls == 'cats':
                color = (random.randint(100, 200), random.randint(50, 100), random.randint(50, 100))
            else:
                color = (random.randint(50, 100), random.randint(100, 200), random.randint(50, 100))

            img = Image.new('RGB', (224, 224), color)
            # Add some noise
            arr = np.array(img)
            noise = np.random.randint(-30, 30, arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

            img.save(os.path.join(class_dir, f'{cls[:-1]}_{i:04d}.jpg'))

    print(f"Sample dataset created at {output_dir}")


def split_dataset(source_dir: str, dest_dir: str,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42):
    """
    Split dataset into train/validation/test sets.
    """
    random.seed(seed)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    splits = ['train', 'val', 'test']
    ratios = [train_ratio, val_ratio, test_ratio]

    # Get class directories
    classes = [d for d in os.listdir(source_dir)
               if os.path.isdir(os.path.join(source_dir, d))]

    if not classes:
        print("No class directories found in source directory")
        return

    print(f"Found classes: {classes}")

    # Create destination directories
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

    # Split each class
    total_files = 0
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)

        n_train = int(len(files) * train_ratio)
        n_val = int(len(files) * val_ratio)

        splits_files = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:]
        }

        for split, split_files in splits_files.items():
            for f in split_files:
                src = os.path.join(class_dir, f)
                dst = os.path.join(dest_dir, split, cls, f)
                shutil.copy2(src, dst)
                total_files += 1

        print(f"  {cls}: train={len(splits_files['train'])}, "
              f"val={len(splits_files['val'])}, test={len(splits_files['test'])}")

    print(f"\nDataset split complete. Total {total_files} files processed.")
    print(f"Output saved to {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Cats vs Dogs Dataset')
    parser.add_argument('--source', type=str, default='data/raw',
                        help='Source data directory')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset from Kaggle')
    parser.add_argument('--sample', action='store_true',
                        help='Create sample dataset for testing')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.source, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    # Download or create sample dataset
    if args.download:
        download_dataset(args.source)
    elif args.sample:
        create_sample_dataset(args.source)

    # Check if source has data
    if not os.listdir(args.source):
        print("Source directory is empty. Creating sample dataset...")
        create_sample_dataset(args.source)

    # Split dataset
    split_dataset(
        source_dir=args.source,
        dest_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
