"""
CNN Model Architecture for Cats vs Dogs Classification
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def create_baseline_cnn(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a baseline CNN model for binary image classification.

    Args:
        input_shape: Tuple of input image dimensions (height, width, channels)
        num_classes: Number of output classes (2 for binary classification)

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Fourth Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def create_simple_cnn(input_shape=(224, 224, 3)):
    """
    Create a simpler CNN for faster training/testing.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def load_model(model_path: str):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model file (.h5)

    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)


def save_model(model, model_path: str):
    """
    Save a trained model to disk.

    Args:
        model: Trained Keras model
        model_path: Path to save the model
    """
    model.save(model_path)
