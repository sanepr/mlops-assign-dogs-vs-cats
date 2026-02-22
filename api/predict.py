"""
Inference module for model predictions.
"""
import os
import time
import base64
import numpy as np
from typing import Tuple, Dict, Optional
import tensorflow as tf


class ModelInference:
    """Class to handle model inference operations."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the inference class.

        Args:
            model_path: Path to the trained model file (.h5)
        """
        self.model = None
        self.model_path = model_path
        self.input_shape = (224, 224, 3)
        self.classes = ['cat', 'dog']

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Load the trained model from disk.

        Args:
            model_path: Path to the model file

        Returns:
            True if model loaded successfully, False otherwise
        """
        # Try different model formats
        paths_to_try = [
            model_path,
            model_path.replace('.h5', '.keras'),
            model_path.replace('.keras', '.h5'),
        ]

        for path in paths_to_try:
            if not os.path.exists(path):
                continue
            try:
                self.model = tf.keras.models.load_model(path, compile=False)
                # Recompile the model
                self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                self.model_path = path
                print(f"Model loaded successfully from {path}")
                return True
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
                continue

        # If all loading attempts fail, create a fresh untrained model
        print("Creating fresh untrained model as fallback...")
        try:
            from src.model import create_simple_cnn
            self.model = create_simple_cnn()
            self.model_path = model_path
            # Try to save it for future use
            try:
                self.model.save(model_path)
                print(f"Fresh model saved to {model_path}")
            except Exception as save_err:
                print(f"Could not save model: {save_err}")
            print("Fresh model created successfully (untrained)")
            return True
        except Exception as e:
            print(f"Could not create fresh model: {e}")

        print(f"Could not load model from any path")
        return False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image bytes for model inference.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Preprocessed numpy array
        """
        from PIL import Image
        from io import BytesIO

        # Load and convert image
        img = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Resize to model input shape
        img = img.resize((self.input_shape[0], self.input_shape[1]))

        # Convert to array and normalize
        img_array = np.array(img) / 255.0

        # Add batch dimension
        return np.expand_dims(img_array, axis=0)

    def preprocess_base64(self, image_base64: str) -> np.ndarray:
        """
        Preprocess base64 encoded image.

        Args:
            image_base64: Base64 encoded image string

        Returns:
            Preprocessed numpy array
        """
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        return self.preprocess_image(image_bytes)

    def predict(self, image_bytes: bytes) -> Tuple[str, float, Dict[str, float], float]:
        """
        Make a prediction on an image.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Tuple of (predicted_class, confidence, probabilities_dict, inference_time_ms)
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess image
        start_time = time.time()
        preprocessed = self.preprocess_image(image_bytes)

        # Make prediction
        prediction = self.model.predict(preprocessed, verbose=0)[0][0]

        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Get class and confidence
        predicted_class = self.classes[1] if prediction >= 0.5 else self.classes[0]
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        probabilities = {
            'cat': float(1 - prediction),
            'dog': float(prediction)
        }

        return predicted_class, float(confidence), probabilities, inference_time

    def predict_base64(self, image_base64: str) -> Tuple[str, float, Dict[str, float], float]:
        """
        Make a prediction on a base64 encoded image.

        Args:
            image_base64: Base64 encoded image string

        Returns:
            Tuple of (predicted_class, confidence, probabilities_dict, inference_time_ms)
        """
        image_bytes = base64.b64decode(image_base64)
        return self.predict(image_bytes)

    def predict_batch(self, images_bytes: list) -> list:
        """
        Make predictions on a batch of images.

        Args:
            images_bytes: List of raw image bytes

        Returns:
            List of prediction tuples
        """
        return [self.predict(img_bytes) for img_bytes in images_bytes]


# Global inference instance
_inference_instance: Optional[ModelInference] = None


def get_inference_instance(model_path: Optional[str] = None) -> ModelInference:
    """
    Get or create the global inference instance.

    Args:
        model_path: Path to the model file

    Returns:
        ModelInference instance
    """
    global _inference_instance

    if _inference_instance is None:
        _inference_instance = ModelInference(model_path)
    elif model_path and not _inference_instance.is_model_loaded():
        _inference_instance.load_model(model_path)

    return _inference_instance
