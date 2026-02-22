"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone


def utc_now():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    timestamp: datetime = Field(default_factory=utc_now)
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint (for base64 input)."""
    image_base64: str = Field(..., description="Base64 encoded image")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: str = Field(..., description="Predicted class label",
                           json_schema_extra={"example": "dog"})
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score",
                             json_schema_extra={"example": 0.95})
    probabilities: dict = Field(..., description="Class probabilities",
                               json_schema_extra={"example": {"cat": 0.05, "dog": 0.95}})
    inference_time_ms: float = Field(..., description="Inference time in milliseconds",
                                    json_schema_extra={"example": 45.2})


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    images_base64: List[str] = Field(..., description="List of base64 encoded images")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_inference_time_ms: float = Field(..., description="Total inference time")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=utc_now)


class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint."""
    model_name: str = Field(..., json_schema_extra={"example": "cats_vs_dogs_classifier"})
    model_version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    input_shape: List[int] = Field(..., json_schema_extra={"example": [224, 224, 3]})
    classes: List[str] = Field(..., json_schema_extra={"example": ["cat", "dog"]})
    framework: str = Field(..., json_schema_extra={"example": "tensorflow"})
