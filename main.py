"""
MLOps Pipeline - Cats vs Dogs Classification
Entry point for running the API server locally
"""
import uvicorn


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == '__main__':
    main()
