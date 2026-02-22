#!/usr/bin/env python3
"""
Register model in MLflow Model Registry
"""
import mlflow
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

model_name = "cats_dogs_classifier"
client = mlflow.MlflowClient()

# Get experiments
print("Fetching experiments...")
experiments = client.search_experiments()
print(f"Found {len(experiments)} experiments")

for exp in experiments:
    print(f"  - {exp.name} (ID: {exp.experiment_id})")

# Find completed runs
print("\nSearching for completed runs...")
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=5
)

print(f"Found {len(runs)} completed runs")

if runs:
    for run in runs:
        run_name = run.data.tags.get('mlflow.runName', 'N/A')
        print(f"  - Run: {run.info.run_id[:8]}... Name: {run_name}")

    # Register the latest completed run
    latest_run = runs[0]
    print(f"\nRegistering model from run: {latest_run.info.run_id}")

    # Check if model artifact exists
    artifacts = client.list_artifacts(latest_run.info.run_id)
    print(f"Artifacts: {[a.path for a in artifacts]}")

    # Try to register the model
    model_uri = f"runs:/{latest_run.info.run_id}/model"
    try:
        result = mlflow.register_model(model_uri, model_name)
        print(f"\n✅ Model registered successfully!")
        print(f"   Name: {model_name}")
        print(f"   Version: {result.version}")
    except Exception as e:
        print(f"\n⚠️ Could not register model: {e}")

        # Try alternative - register from local model file
        print("\nTrying to register from local model file...")
        model_path = "models/baseline_model.h5"
        if os.path.exists(model_path):
            print(f"Found model at {model_path}")
            # Log and register using MLflow
            with mlflow.start_run(run_name="model_registration"):
                mlflow.log_artifact(model_path)
                mlflow.log_param("model_type", "baseline_cnn")
                mlflow.log_param("framework", "tensorflow/keras")
                print("Model artifact logged to MLflow")
else:
    print("No completed runs found. Current training may still be in progress.")

    # Register existing local model
    print("\nRegistering local model...")
    model_path = "models/baseline_model.h5"
    if os.path.exists(model_path):
        print(f"Found model at {model_path}")
        mlflow.set_experiment("cats_vs_dogs")
        with mlflow.start_run(run_name="local_model_registration"):
            mlflow.log_artifact(model_path)
            mlflow.log_param("model_type", "baseline_cnn")
            mlflow.log_param("source", "local")
            print("✅ Local model registered to MLflow")
