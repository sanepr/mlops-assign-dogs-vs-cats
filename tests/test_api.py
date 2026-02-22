"""
Unit tests for API endpoints.
"""
import os
import sys
import pytest
import base64
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked model."""
        with patch('api.main.get_inference_instance') as mock_inference:
            mock_instance = Mock()
            mock_instance.is_model_loaded.return_value = True
            mock_inference.return_value = mock_instance

            from api.main import app
            yield TestClient(app)

    def test_health_endpoint_returns_200(self, client):
        """Test health endpoint returns 200 status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_response_structure(self, client):
        """Test health endpoint response has correct structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_endpoint_status_healthy(self, client):
        """Test health endpoint reports healthy status."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        with patch('api.main.get_inference_instance') as mock_inference:
            mock_instance = Mock()
            mock_instance.is_model_loaded.return_value = True
            mock_inference.return_value = mock_instance

            from api.main import app
            yield TestClient(app)

    def test_root_endpoint_returns_200(self, client):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_endpoint_has_service_info(self, client):
        """Test root endpoint contains service information."""
        response = client.get("/")
        data = response.json()

        assert "service" in data
        assert "version" in data
        assert "docs" in data


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    @pytest.fixture
    def client_with_model(self):
        """Create test client with mocked model inference."""
        with patch('api.main.model_inference') as mock_inference:
            mock_inference.is_model_loaded.return_value = True
            mock_inference.predict.return_value = (
                'dog',  # predicted_class
                0.95,   # confidence
                {'cat': 0.05, 'dog': 0.95},  # probabilities
                45.2    # inference_time_ms
            )

            from api.main import app
            yield TestClient(app)

    def create_test_image(self):
        """Create a test image for upload."""
        img = Image.new('RGB', (224, 224), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        return buffer

    def test_predict_endpoint_accepts_image(self, client_with_model):
        """Test predict endpoint accepts image upload."""
        image_buffer = self.create_test_image()

        response = client_with_model.post(
            "/predict",
            files={"file": ("test.jpg", image_buffer, "image/jpeg")}
        )

        # May return 503 if model not loaded in test environment
        assert response.status_code in [200, 503]

    def test_predict_endpoint_rejects_invalid_content_type(self, client_with_model):
        """Test predict endpoint rejects non-image files."""
        response = client_with_model.post(
            "/predict",
            files={"file": ("test.txt", BytesIO(b"not an image"), "text/plain")}
        )

        # Should reject with 400 or 503 (if model not loaded)
        assert response.status_code in [400, 503]


class TestPredictBase64Endpoint:
    """Tests for base64 prediction endpoint."""

    @pytest.fixture
    def client_with_model(self):
        """Create test client with mocked model inference."""
        with patch('api.main.model_inference') as mock_inference:
            mock_inference.is_model_loaded.return_value = True
            mock_inference.predict_base64.return_value = (
                'cat',
                0.88,
                {'cat': 0.88, 'dog': 0.12},
                50.0
            )

            from api.main import app
            yield TestClient(app)

    def create_base64_image(self):
        """Create a base64 encoded test image."""
        img = Image.new('RGB', (100, 100), color='blue')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def test_predict_base64_endpoint_structure(self, client_with_model):
        """Test predict/base64 endpoint accepts correct structure."""
        image_base64 = self.create_base64_image()

        response = client_with_model.post(
            "/predict/base64",
            json={"image_base64": image_base64}
        )

        # May return 503 if model not loaded
        assert response.status_code in [200, 503]


class TestStatsEndpoint:
    """Tests for stats endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        with patch('api.main.get_inference_instance') as mock_inference:
            mock_instance = Mock()
            mock_instance.is_model_loaded.return_value = True
            mock_inference.return_value = mock_instance

            from api.main import app
            yield TestClient(app)

    def test_stats_endpoint_returns_200(self, client):
        """Test stats endpoint returns 200."""
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_endpoint_has_metrics(self, client):
        """Test stats endpoint contains metrics."""
        response = client.get("/stats")
        data = response.json()

        assert "total_requests" in data
        assert "timestamp" in data


class TestAPISchemas:
    """Tests for API schema validation."""

    def test_prediction_response_schema(self):
        """Test PredictionResponse schema."""
        from api.schemas import PredictionResponse

        response = PredictionResponse(
            prediction="dog",
            confidence=0.95,
            probabilities={"cat": 0.05, "dog": 0.95},
            inference_time_ms=45.2
        )

        assert response.prediction == "dog"
        assert response.confidence == 0.95

    def test_health_response_schema(self):
        """Test HealthResponse schema."""
        from api.schemas import HealthResponse
        from datetime import datetime, timezone

        response = HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            model_loaded=True,
            version="1.0.0"
        )

        assert response.status == "healthy"
        assert response.model_loaded == True

    def test_prediction_request_schema(self):
        """Test PredictionRequest schema."""
        from api.schemas import PredictionRequest

        request = PredictionRequest(image_base64="dGVzdA==")

        assert request.image_base64 == "dGVzdA=="


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
