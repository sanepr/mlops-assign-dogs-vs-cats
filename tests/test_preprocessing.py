"""
Unit tests for data preprocessing functions.
"""
import os
import sys
import pytest
import numpy as np
from PIL import Image
from io import BytesIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    preprocess_image_bytes,
    validate_image,
    get_class_label,
    IMAGE_SIZE
)


class TestPreprocessImageBytes:
    """Tests for preprocess_image_bytes function."""

    def create_test_image(self, width=300, height=300, color='RGB'):
        """Helper to create a test image."""
        img = Image.new(color, (width, height), color=(255, 0, 0))
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()

    def test_preprocess_image_bytes_shape(self):
        """Test that preprocessing returns correct shape."""
        image_bytes = self.create_test_image()
        result = preprocess_image_bytes(image_bytes)

        assert result.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    def test_preprocess_image_bytes_normalization(self):
        """Test that pixel values are normalized to [0, 1]."""
        image_bytes = self.create_test_image()
        result = preprocess_image_bytes(image_bytes)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_image_bytes_dtype(self):
        """Test that result is float type."""
        image_bytes = self.create_test_image()
        result = preprocess_image_bytes(image_bytes)

        assert result.dtype == np.float64 or result.dtype == np.float32

    def test_preprocess_image_bytes_different_sizes(self):
        """Test preprocessing handles different input sizes."""
        for size in [(100, 100), (500, 300), (224, 224), (1000, 800)]:
            image_bytes = self.create_test_image(size[0], size[1])
            result = preprocess_image_bytes(image_bytes)

            assert result.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    def test_preprocess_image_bytes_grayscale_converted(self):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale image
        img = Image.new('L', (300, 300), color=128)
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

        result = preprocess_image_bytes(image_bytes)

        # Should still have 3 channels after RGB conversion
        assert result.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


class TestValidateImage:
    """Tests for validate_image function."""

    def create_valid_image(self):
        """Create valid image bytes."""
        img = Image.new('RGB', (100, 100), color='blue')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()

    def test_validate_image_valid_jpeg(self):
        """Test validation of valid JPEG image."""
        image_bytes = self.create_valid_image()
        assert validate_image(image_bytes) == True

    def test_validate_image_valid_png(self):
        """Test validation of valid PNG image."""
        img = Image.new('RGB', (100, 100), color='green')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        assert validate_image(image_bytes) == True

    def test_validate_image_invalid_bytes(self):
        """Test validation of invalid image bytes."""
        invalid_bytes = b'not an image'
        assert validate_image(invalid_bytes) == False

    def test_validate_image_empty_bytes(self):
        """Test validation of empty bytes."""
        assert validate_image(b'') == False

    def test_validate_image_truncated_image(self):
        """Test validation of truncated image."""
        valid_image = self.create_valid_image()
        truncated = valid_image[:len(valid_image)//2]

        # Truncated images may or may not pass validation depending on the truncation point
        result = validate_image(truncated)
        assert isinstance(result, bool)


class TestGetClassLabel:
    """Tests for get_class_label function."""

    def test_get_class_label_dog_high_confidence(self):
        """Test classification as dog with high confidence."""
        assert get_class_label(0.9) == 'dog'

    def test_get_class_label_cat_high_confidence(self):
        """Test classification as cat with high confidence."""
        assert get_class_label(0.1) == 'cat'

    def test_get_class_label_threshold_exactly_0_5(self):
        """Test classification at exactly threshold."""
        assert get_class_label(0.5) == 'dog'  # >= threshold is dog

    def test_get_class_label_just_below_threshold(self):
        """Test classification just below threshold."""
        assert get_class_label(0.49) == 'cat'

    def test_get_class_label_custom_threshold(self):
        """Test classification with custom threshold."""
        assert get_class_label(0.6, threshold=0.7) == 'cat'
        assert get_class_label(0.8, threshold=0.7) == 'dog'

    def test_get_class_label_boundary_values(self):
        """Test classification at boundary values."""
        assert get_class_label(0.0) == 'cat'
        assert get_class_label(1.0) == 'dog'


class TestImagePreprocessingEdgeCases:
    """Edge case tests for image preprocessing."""

    def test_preprocess_very_small_image(self):
        """Test preprocessing of very small image."""
        img = Image.new('RGB', (10, 10), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

        result = preprocess_image_bytes(image_bytes)
        assert result.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    def test_preprocess_non_square_image(self):
        """Test preprocessing of non-square image."""
        img = Image.new('RGB', (100, 500), color='green')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

        result = preprocess_image_bytes(image_bytes)
        assert result.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    def test_preprocess_rgba_image(self):
        """Test preprocessing of RGBA image (with alpha channel)."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        result = preprocess_image_bytes(image_bytes)
        # Should be converted to RGB (3 channels)
        assert result.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
