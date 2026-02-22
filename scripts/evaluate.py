"""
Model Evaluation Script
Evaluates the trained model and generates metrics/visualizations
"""
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_model(model_path: str, data_dir: str, output_dir: str = 'models'):
    """
    Evaluate the trained model on test data.
    """
    import tensorflow as tf
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_curve, auc, precision_recall_curve
    )
    from src.utils import create_test_generator

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading test data from {data_dir}...")
    test_gen = create_test_generator(data_dir)

    print("Making predictions...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred_prob = predictions.flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = test_gen.classes

    # Calculate metrics
    print("Calculating metrics...")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    report = classification_report(y_true, y_pred,
                                   target_names=['cat', 'dog'],
                                   output_dict=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)

    # Compile metrics
    metrics = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model_path': model_path,
        'test_samples': len(y_true),
        'accuracy': float(report['accuracy']),
        'cat': {
            'precision': float(report['cat']['precision']),
            'recall': float(report['cat']['recall']),
            'f1-score': float(report['cat']['f1-score']),
            'support': int(report['cat']['support'])
        },
        'dog': {
            'precision': float(report['dog']['precision']),
            'recall': float(report['dog']['recall']),
            'f1-score': float(report['dog']['f1-score']),
            'support': int(report['dog']['support'])
        },
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist()
    }

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Cat', 'Dog'],
                    yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")

        # ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {roc_path}")

        # Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
        plt.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Precision-Recall curve saved to {pr_path}")

    except ImportError as e:
        print(f"Could not generate plots: {e}")

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Test Samples: {metrics['test_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"\nCat - Precision: {metrics['cat']['precision']:.4f}, "
          f"Recall: {metrics['cat']['recall']:.4f}, "
          f"F1: {metrics['cat']['f1-score']:.4f}")
    print(f"Dog - Precision: {metrics['dog']['precision']:.4f}, "
          f"Recall: {metrics['dog']['recall']:.4f}, "
          f"F1: {metrics['dog']['f1-score']:.4f}")
    print("="*50)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for metrics and plots')

    args = parser.parse_args()

    evaluate_model(args.model, args.data, args.output)


if __name__ == '__main__':
    main()
