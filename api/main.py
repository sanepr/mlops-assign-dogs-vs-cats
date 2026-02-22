"""
FastAPI Application for Cats vs Dogs Classification Service

This module provides a REST API for serving the trained CNN model
for binary image classification (cats vs dogs).
"""
import os
import time
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import (
    HealthResponse, PredictionResponse, PredictionRequest,
    ErrorResponse, ModelInfoResponse, BatchPredictionRequest,
    BatchPredictionResponse
)
from api.predict import ModelInference, get_inference_instance


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Application configuration
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/baseline_model.h5')
APP_VERSION = os.environ.get('APP_VERSION', '1.0.0')
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

# Global model instance
model_inference: Optional[ModelInference] = None

# Request metrics
request_count = 0
total_inference_time = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global model_inference

    # Startup
    logger.info("Starting up application", model_path=MODEL_PATH)

    try:
        model_inference = get_inference_instance(MODEL_PATH)
        if model_inference.is_model_loaded():
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model not loaded - predictions will fail until model is available")
    except Exception as e:
        logger.error("Failed to load model", error=str(e))

    yield

    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI application
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="A REST API for binary image classification (Cats vs Dogs) using a CNN model",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[],
    inprogress_name="inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their responses."""
    global request_count, total_inference_time

    start_time = time.time()
    request_count += 1

    # Log request (excluding sensitive data)
    logger.info(
        "Request received",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown",
        request_id=request_count
    )

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000

    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time_ms=round(process_time, 2),
        request_id=request_count
    )

    response.headers["X-Process-Time"] = str(round(process_time, 2))
    response.headers["X-Request-ID"] = str(request_count)

    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error("Unhandled exception", error=str(exc), url=str(request.url))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if DEBUG else None
        ).model_dump()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service status.

    Returns:
        HealthResponse with current service status
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        model_loaded=model_inference.is_model_loaded() if model_inference else False,
        version=APP_VERSION
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Cats vs Dogs Classification API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# Model info endpoint
@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        ModelInfoResponse with model details
    """
    if not model_inference or not model_inference.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name="cats_vs_dogs_classifier",
        model_version=APP_VERSION,
        input_shape=list(model_inference.input_shape),
        classes=model_inference.classes,
        framework="tensorflow"
    )


# Prediction endpoint (file upload)
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Image file (JPEG, JPG, PNG)"
                            }
                        },
                        "required": ["file"]
                    },
                    "encoding": {
                        "file": {
                            "contentType": "image/jpeg, image/png, image/jpg"
                        }
                    }
                }
            }
        }
    }
)
async def predict(
    file: UploadFile = File(..., description="Image file to classify (JPEG, JPG, or PNG format)")
):
    """
    Make a prediction on an uploaded image.

    Accepts image files in JPEG, JPG, or PNG format.

    Args:
        file: Image file (JPEG, JPG, PNG)

    Returns:
        PredictionResponse with class prediction and confidence
    """
    global total_inference_time

    if not model_inference or not model_inference.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type - check both content type and file extension
    allowed_content_types = ["image/jpeg", "image/png", "image/jpg", "application/octet-stream"]
    allowed_extensions = ['.jpg', '.jpeg', '.png']

    filename_lower = (file.filename or "").lower()
    has_valid_extension = any(filename_lower.endswith(ext) for ext in allowed_extensions)
    has_valid_content_type = file.content_type in allowed_content_types

    # Accept if either content type OR extension is valid
    if not (has_valid_content_type or has_valid_extension):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Supported formats: JPEG, JPG, PNG"
        )

    try:
        # Read and process image
        image_bytes = await file.read()

        # Make prediction
        predicted_class, confidence, probabilities, inference_time = model_inference.predict(image_bytes)

        total_inference_time += inference_time

        logger.info(
            "Prediction made",
            filename=file.filename,
            predicted_class=predicted_class,
            confidence=confidence,
            inference_time_ms=inference_time
        )

        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=round(inference_time, 2)
        )

    except Exception as e:
        logger.error("Prediction failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Prediction endpoint (base64)
@app.post("/predict/base64", response_model=PredictionResponse, tags=["Prediction"])
async def predict_base64(request: PredictionRequest):
    """
    Make a prediction on a base64 encoded image.

    Args:
        request: PredictionRequest with base64 encoded image

    Returns:
        PredictionResponse with class prediction and confidence
    """
    global total_inference_time

    if not model_inference or not model_inference.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Make prediction
        predicted_class, confidence, probabilities, inference_time = model_inference.predict_base64(
            request.image_base64
        )

        total_inference_time += inference_time

        logger.info(
            "Prediction made (base64)",
            predicted_class=predicted_class,
            confidence=confidence,
            inference_time_ms=inference_time
        )

        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=round(inference_time, 2)
        )

    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    files: list[UploadFile] = File(
        ...,
        description="List of image files to classify (JPEG, JPG, or PNG format)"
    )
):
    """
    Make predictions on multiple images.

    Accepts image files in JPEG, JPG, or PNG format.

    Args:
        files: List of image files

    Returns:
        BatchPredictionResponse with list of predictions
    """
    if not model_inference or not model_inference.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    predictions = []

    allowed_content_types = ["image/jpeg", "image/png", "image/jpg", "application/octet-stream"]
    allowed_extensions = ['.jpg', '.jpeg', '.png']

    for file in files:
        filename_lower = (file.filename or "").lower()
        has_valid_extension = any(filename_lower.endswith(ext) for ext in allowed_extensions)
        has_valid_content_type = file.content_type in allowed_content_types

        if not (has_valid_content_type or has_valid_extension):
            continue

        try:
            image_bytes = await file.read()
            predicted_class, confidence, probabilities, inference_time = model_inference.predict(image_bytes)

            predictions.append(PredictionResponse(
                prediction=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                inference_time_ms=round(inference_time, 2)
            ))
        except Exception as e:
            logger.error("Batch prediction failed for file", filename=file.filename, error=str(e))

    total_time = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_inference_time_ms=round(total_time, 2)
    )


# Metrics endpoint
@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """
    Get service statistics.

    Returns:
        Dictionary with service metrics
    """
    return {
        "total_requests": request_count,
        "total_inference_time_ms": round(total_inference_time, 2),
        "average_inference_time_ms": round(total_inference_time / max(request_count, 1), 2),
        "model_loaded": model_inference.is_model_loaded() if model_inference else False,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG
    )
