"""
Model Training Script with MLflow Experiment Tracking

This script trains a CNN model for cats vs dogs classification and logs
all experiments to MLflow.
"""
import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_baseline_cnn, create_simple_cnn, save_model
from src.utils import create_data_generators, create_test_generator


def train_model(data_dir: str,
                output_dir: str = 'models',
                epochs: int = 20,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                model_type: str = 'baseline',
                experiment_name: str = 'cats_vs_dogs'):
    """
    Train the model with MLflow tracking.

    Args:
        data_dir: Path to training data directory
        output_dir: Path to save model artifacts
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        model_type: 'baseline' or 'simple'
        experiment_name: MLflow experiment name
    """
    # Set up MLflow tracking URI (use environment variable or default to local server)
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5001')
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"MLflow tracking URI: {mlflow_uri}")

    # Set up MLflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("input_shape", "224x224x3")

        # Create data generators
        print("Creating data generators...")
        train_gen, val_gen = create_data_generators(
            data_dir,
            batch_size=batch_size,
            validation_split=0.2
        )

        mlflow.log_param("train_samples", train_gen.samples)
        mlflow.log_param("val_samples", val_gen.samples)
        mlflow.log_param("class_indices", json.dumps(train_gen.class_indices))

        # Create model
        print(f"Creating {model_type} model...")
        if model_type == 'baseline':
            model = create_baseline_cnn()
        else:
            model = create_simple_cnn()

        # Update optimizer with specified learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Log model summary
        model.summary()
        mlflow.log_param("total_params", model.count_params())

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        # Train model
        print("Starting training...")
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        # Log metrics for each epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

        # Log final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)

        print(f"\nTraining completed!")
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")

        # Save training history plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Accuracy plot
            axes[0].plot(history.history['accuracy'], label='Train')
            axes[0].plot(history.history['val_accuracy'], label='Validation')
            axes[0].set_title('Model Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()

            # Loss plot
            axes[1].plot(history.history['loss'], label='Train')
            axes[1].plot(history.history['val_loss'], label='Validation')
            axes[1].set_title('Model Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()

            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'training_curves.png')
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
        except ImportError:
            print("Matplotlib not available, skipping plot generation")

        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'{model_type}_model.h5')
        save_model(model, model_path)
        print(f"Model saved to {model_path}")

        # Log model to MLflow
        mlflow.keras.log_model(model, "model")
        mlflow.log_artifact(model_path)

        # Generate and log confusion matrix if validation data available
        print("Evaluating model...")
        val_gen.reset()
        predictions = model.predict(val_gen, verbose=0)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = val_gen.classes

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred,
                                       target_names=['cat', 'dog'],
                                       output_dict=True)

        # Log classification metrics
        mlflow.log_metric("precision_cat", report['cat']['precision'])
        mlflow.log_metric("recall_cat", report['cat']['recall'])
        mlflow.log_metric("f1_cat", report['cat']['f1-score'])
        mlflow.log_metric("precision_dog", report['dog']['precision'])
        mlflow.log_metric("recall_dog", report['dog']['recall'])
        mlflow.log_metric("f1_dog", report['dog']['f1-score'])

        # Save confusion matrix
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Cat', 'Dog'],
                        yticklabels=['Cat', 'Dog'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()
        except ImportError:
            print("Seaborn not available, skipping confusion matrix plot")

        print(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=['cat', 'dog'])}")

        return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs Classifier')
    parser.add_argument('--data-dir', type=str, default='data/train',
                        help='Path to training data directory')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Path to save model artifacts')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--model-type', type=str, default='baseline',
                        choices=['baseline', 'simple'],
                        help='Model architecture to use')
    parser.add_argument('--experiment-name', type=str, default='cats_vs_dogs',
                        help='MLflow experiment name')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train model
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        experiment_name=args.experiment_name
    )


if __name__ == '__main__':
    main()
