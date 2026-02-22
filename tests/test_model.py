"""
Unit tests for model architecture and inference functions.
"""
import os
import sys
import pytest
import numpy as np
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_baseline_cnn, create_simple_cnn, save_model, load_model


class TestModelArchitecture:
    """Tests for model architecture functions."""

    def test_create_baseline_cnn_output_shape(self):
        """Test that baseline CNN has correct output shape."""
        model = create_baseline_cnn()

        # Check output shape (should be (None, 1) for binary classification)
        assert model.output_shape == (None, 1)

    def test_create_baseline_cnn_input_shape(self):
        """Test that baseline CNN has correct input shape."""
        model = create_baseline_cnn()

        # Check input shape
        assert model.input_shape == (None, 224, 224, 3)

    def test_create_baseline_cnn_custom_input_shape(self):
        """Test baseline CNN with custom input shape."""
        custom_shape = (128, 128, 3)
        model = create_baseline_cnn(input_shape=custom_shape)

        assert model.input_shape == (None, 128, 128, 3)

    def test_create_simple_cnn_output_shape(self):
        """Test that simple CNN has correct output shape."""
        model = create_simple_cnn()

        assert model.output_shape == (None, 1)

    def test_create_simple_cnn_input_shape(self):
        """Test that simple CNN has correct input shape."""
        model = create_simple_cnn()

        assert model.input_shape == (None, 224, 224, 3)

    def test_model_is_compiled(self):
        """Test that created models are compiled."""
        baseline = create_baseline_cnn()
        simple = create_simple_cnn()

        # Check optimizer is set (indicates model is compiled)
        assert baseline.optimizer is not None
        assert simple.optimizer is not None

    def test_model_has_layers(self):
        """Test that models have expected layer types."""
        model = create_baseline_cnn()

        layer_types = [type(layer).__name__ for layer in model.layers]

        assert 'Conv2D' in layer_types
        assert 'Dense' in layer_types
        assert 'Flatten' in layer_types


class TestModelPrediction:
    """Tests for model prediction functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return create_simple_cnn()

    def test_model_predict_single_image(self, simple_model):
        """Test prediction on a single image."""
        # Create dummy input
        dummy_input = np.random.random((1, 224, 224, 3))

        prediction = simple_model.predict(dummy_input, verbose=0)

        assert prediction.shape == (1, 1)
        assert 0 <= prediction[0][0] <= 1

    def test_model_predict_batch(self, simple_model):
        """Test prediction on a batch of images."""
        batch_size = 4
        dummy_input = np.random.random((batch_size, 224, 224, 3))

        predictions = simple_model.predict(dummy_input, verbose=0)

        assert predictions.shape == (batch_size, 1)
        assert all(0 <= p <= 1 for p in predictions.flatten())

    def test_model_predict_normalized_input(self, simple_model):
        """Test prediction with properly normalized input."""
        # Input normalized to [0, 1]
        dummy_input = np.random.random((1, 224, 224, 3))

        prediction = simple_model.predict(dummy_input, verbose=0)

        # Should produce valid probability
        assert 0 <= prediction[0][0] <= 1


class TestModelSaveLoad:
    """Tests for model save and load functionality."""

    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        model = create_simple_cnn()

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            model_path = f.name

        try:
            # Save model
            save_model(model, model_path)
            assert os.path.exists(model_path)

            # Load model
            loaded_model = load_model(model_path)
            assert loaded_model is not None

            # Compare predictions
            dummy_input = np.random.random((1, 224, 224, 3))
            original_pred = model.predict(dummy_input, verbose=0)
            loaded_pred = loaded_model.predict(dummy_input, verbose=0)

            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_load_nonexistent_model(self):
        """Test loading a non-existent model raises error."""
        with pytest.raises(Exception):
            load_model('/nonexistent/path/model.h5')


class TestModelTrainingReadiness:
    """Tests to verify model is ready for training."""

    def test_model_can_compute_loss(self):
        """Test that model can compute loss on dummy data."""
        model = create_simple_cnn()

        # Create dummy data
        x = np.random.random((2, 224, 224, 3))
        y = np.array([[0], [1]])

        # Evaluate should work without errors
        loss = model.evaluate(x, y, verbose=0)

        assert isinstance(loss, list) or isinstance(loss, float)

    def test_model_can_train_one_step(self):
        """Test that model can perform one training step."""
        model = create_simple_cnn()

        # Create dummy data
        x = np.random.random((2, 224, 224, 3))
        y = np.array([[0], [1]])

        # Train for one epoch
        history = model.fit(x, y, epochs=1, verbose=0)

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
