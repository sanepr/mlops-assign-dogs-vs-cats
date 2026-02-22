#!/usr/bin/env python3
"""Script to verify and display MLflow experiments"""
import os
import subprocess
import sqlite3
import time

PROJECT_DIR = '/Users/aashishr/codebase/mlso_ass'
MLRUNS_DIR = os.path.join(PROJECT_DIR, 'mlruns')
MLFLOW_DB = os.path.join(MLRUNS_DIR, 'mlflow.db')

print("=" * 60)
print("MLflow Experiments Verification")
print("=" * 60)

# Check if mlflow.db exists
if os.path.exists(MLFLOW_DB):
    print(f"\n✓ MLflow database found: {MLFLOW_DB}")
    print(f"  Size: {os.path.getsize(MLFLOW_DB)} bytes")

    # Query experiments from SQLite
    try:
        conn = sqlite3.connect(MLFLOW_DB)
        cursor = conn.cursor()

        # Get experiments
        print("\n--- Experiments ---")
        cursor.execute("SELECT experiment_id, name, lifecycle_stage FROM experiments")
        experiments = cursor.fetchall()
        for exp in experiments:
            print(f"  ID: {exp[0]}, Name: {exp[1]}, Stage: {exp[2]}")

        # Get runs
        print("\n--- Runs ---")
        cursor.execute("""
            SELECT run_uuid, name, status, start_time, end_time 
            FROM runs 
            ORDER BY start_time DESC 
            LIMIT 10
        """)
        runs = cursor.fetchall()
        for run in runs:
            print(f"  Run: {run[0][:8]}..., Name: {run[1]}, Status: {run[2]}")

        # Get metrics
        print("\n--- Latest Metrics ---")
        cursor.execute("""
            SELECT r.name, m.key, m.value 
            FROM latest_metrics m 
            JOIN runs r ON m.run_uuid = r.run_uuid
            ORDER BY r.start_time DESC
            LIMIT 20
        """)
        metrics = cursor.fetchall()
        for m in metrics:
            print(f"  {m[0]}: {m[1]} = {m[2]}")

        conn.close()

    except Exception as e:
        print(f"  Error reading database: {e}")
else:
    print(f"\n✗ MLflow database NOT found at: {MLFLOW_DB}")

# Check mlruns folder structure
print("\n--- MLruns Directory Structure ---")
if os.path.exists(MLRUNS_DIR):
    for item in os.listdir(MLRUNS_DIR):
        item_path = os.path.join(MLRUNS_DIR, item)
        if os.path.isdir(item_path):
            count = len(os.listdir(item_path))
            print(f"  {item}/ ({count} items)")
        else:
            print(f"  {item}")

print("\n" + "=" * 60)
print("To view experiments in MLflow UI:")
print("  1. Ensure Docker is running: docker-compose up -d")
print("  2. Open: http://localhost:5001")
print("=" * 60)
