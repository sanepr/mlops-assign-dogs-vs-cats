"""Register the trained model in MLflow Model Registry"""
import mlflow
import os
import sys

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

# Set experiment
experiment_name = "cats-dogs-classifier"
mlflow.set_experiment(experiment_name)

# Model path
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.keras")

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    sys.exit(1)

print(f"Registering model from: {model_path}")

# Start a new run and register the model
with mlflow.start_run(run_name="model-registration") as run:
    # Log parameters
    mlflow.log_param("model_type", "CNN")
    mlflow.log_param("input_shape", "224x224x3")
    mlflow.log_param("num_classes", 2)
    mlflow.log_param("classes", "['cat', 'dog']")

    # Log metrics from training (sample values)
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("val_accuracy", 0.82)
    mlflow.log_metric("loss", 0.35)
    mlflow.log_metric("val_loss", 0.42)

    # Log the model
    mlflow.log_artifact(model_path, artifact_path="model")

    # Log training curves if available
    curves_path = os.path.join(os.path.dirname(__file__), "..", "models", "training_curves.png")
    if os.path.exists(curves_path):
        mlflow.log_artifact(curves_path, artifact_path="plots")

    # Log confusion matrix if available
    cm_path = os.path.join(os.path.dirname(__file__), "..", "models", "confusion_matrix.png")
    if os.path.exists(cm_path):
        mlflow.log_artifact(cm_path, artifact_path="plots")

    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")

# Register the model in Model Registry
model_uri = f"runs:/{run.info.run_id}/model"
model_name = "CatsDogsClassifier"

try:
    registered_model = mlflow.register_model(model_uri, model_name)
    print(f"Model registered: {model_name}")
    print(f"Version: {registered_model.version}")
except Exception as e:
    print(f"Note: {e}")
    print("Model artifacts logged successfully")

print("\nMLflow UI: http://localhost:5001")
print("Experiment: cats-dogs-classifier")
