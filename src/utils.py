"""
Utility functions for data loading, preprocessing, and augmentation.
"""
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def preprocess_image(image_path: str, target_size=IMAGE_SIZE) -> np.ndarray:
    """
    Load and preprocess a single image for model inference.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)

    Returns:
        Preprocessed numpy array of shape (1, height, width, 3)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)


def preprocess_image_bytes(image_bytes: bytes, target_size=IMAGE_SIZE) -> np.ndarray:
    """
    Preprocess image from bytes for inference.

    Args:
        image_bytes: Raw image bytes
        target_size: Target size for resizing (height, width)

    Returns:
        Preprocessed numpy array of shape (1, height, width, 3)
    """
    from io import BytesIO
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def create_data_generators(data_dir: str,
                           validation_split: float = 0.2,
                           batch_size: int = BATCH_SIZE,
                           target_size=IMAGE_SIZE):
    """
    Create training and validation data generators with augmentation.

    Args:
        data_dir: Path to the data directory with class subdirectories
        validation_split: Fraction of data for validation
        batch_size: Batch size for training
        target_size: Target image size

    Returns:
        Tuple of (train_generator, validation_generator)
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )

    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator


def create_test_generator(test_dir: str,
                          batch_size: int = BATCH_SIZE,
                          target_size=IMAGE_SIZE):
    """
    Create test data generator.

    Args:
        test_dir: Path to test data directory
        batch_size: Batch size
        target_size: Target image size

    Returns:
        Test data generator
    """
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return test_generator


def split_dataset(source_dir: str, dest_dir: str,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1):
    """
    Split dataset into train/validation/test sets.

    Args:
        source_dir: Source directory with class subdirectories
        dest_dir: Destination directory for split data
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    """
    import shutil
    import random

    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    splits = ['train', 'val', 'test']
    ratios = [train_ratio, val_ratio, test_ratio]

    # Get class directories
    classes = [d for d in os.listdir(source_dir)
               if os.path.isdir(os.path.join(source_dir, d))]

    # Create destination directories
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

    # Split each class
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        files = os.listdir(class_dir)
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

    print(f"Dataset split complete. Files saved to {dest_dir}")


def validate_image(image_bytes: bytes) -> bool:
    """
    Validate that the image bytes are a valid image.

    Args:
        image_bytes: Raw image bytes

    Returns:
        True if valid image, False otherwise
    """
    try:
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
        img.verify()
        return True
    except Exception:
        return False


def get_class_label(prediction: float, threshold: float = 0.5) -> str:
    """
    Convert model prediction to class label.

    Args:
        prediction: Model prediction probability
        threshold: Classification threshold

    Returns:
        Class label ('cat' or 'dog')
    """
    return 'dog' if prediction >= threshold else 'cat'
