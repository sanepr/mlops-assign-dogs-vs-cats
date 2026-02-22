"""
Model Performance Tracking Script
Simulates real-world predictions and tracks model performance over time
"""
import os
import sys
import json
import random
import time
import argparse
from datetime import datetime, timezone
from typing import List, Dict
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PerformanceTracker:
    """Track model performance with real or simulated predictions."""

    def __init__(self, api_url: str = "http://localhost:8000", log_dir: str = "logs"):
        self.api_url = api_url
        self.log_dir = log_dir
        self.predictions_log: List[Dict] = []
        os.makedirs(log_dir, exist_ok=True)

    def simulate_prediction(self) -> Dict:
        """
        Simulate a prediction request.
        In production, this would be real user requests.
        """
        # Simulated true labels (ground truth)
        true_label = random.choice(['cat', 'dog'])

        # Simulated model prediction (with ~90% accuracy)
        if random.random() < 0.9:
            predicted_label = true_label
        else:
            predicted_label = 'dog' if true_label == 'cat' else 'cat'

        # Simulated confidence
        confidence = random.uniform(0.7, 0.99) if predicted_label == true_label else random.uniform(0.5, 0.7)

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': round(confidence, 4),
            'correct': true_label == predicted_label,
            'inference_time_ms': round(random.uniform(30, 80), 2)
        }

    def collect_predictions(self, n_samples: int = 100) -> List[Dict]:
        """Collect a batch of predictions."""
        print(f"Collecting {n_samples} predictions...")

        predictions = []
        for i in range(n_samples):
            pred = self.simulate_prediction()
            predictions.append(pred)
            self.predictions_log.append(pred)

            if (i + 1) % 20 == 0:
                print(f"  Collected {i + 1}/{n_samples} predictions")

        return predictions

    def calculate_metrics(self, predictions: List[Dict] = None) -> Dict:
        """Calculate performance metrics from predictions."""
        if predictions is None:
            predictions = self.predictions_log

        if not predictions:
            return {}

        # Basic metrics
        total = len(predictions)
        correct = sum(1 for p in predictions if p['correct'])
        accuracy = correct / total

        # Per-class metrics
        cat_preds = [p for p in predictions if p['true_label'] == 'cat']
        dog_preds = [p for p in predictions if p['true_label'] == 'dog']

        cat_correct = sum(1 for p in cat_preds if p['correct'])
        dog_correct = sum(1 for p in dog_preds if p['correct'])

        # Confusion matrix
        tp_cat = sum(1 for p in predictions if p['true_label'] == 'cat' and p['predicted_label'] == 'cat')
        fp_cat = sum(1 for p in predictions if p['true_label'] == 'dog' and p['predicted_label'] == 'cat')
        fn_cat = sum(1 for p in predictions if p['true_label'] == 'cat' and p['predicted_label'] == 'dog')
        tp_dog = sum(1 for p in predictions if p['true_label'] == 'dog' and p['predicted_label'] == 'dog')

        precision_cat = tp_cat / (tp_cat + fp_cat) if (tp_cat + fp_cat) > 0 else 0
        recall_cat = tp_cat / (tp_cat + fn_cat) if (tp_cat + fn_cat) > 0 else 0

        precision_dog = tp_dog / (tp_dog + fn_cat) if (tp_dog + fn_cat) > 0 else 0
        recall_dog = tp_dog / (tp_dog + fp_cat) if (tp_dog + fp_cat) > 0 else 0

        # Latency metrics
        latencies = [p['inference_time_ms'] for p in predictions]
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        # Average confidence
        avg_confidence = sum(p['confidence'] for p in predictions) / total

        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_predictions': total,
            'accuracy': round(accuracy, 4),
            'cat': {
                'total': len(cat_preds),
                'correct': cat_correct,
                'accuracy': round(cat_correct / len(cat_preds), 4) if cat_preds else 0,
                'precision': round(precision_cat, 4),
                'recall': round(recall_cat, 4)
            },
            'dog': {
                'total': len(dog_preds),
                'correct': dog_correct,
                'accuracy': round(dog_correct / len(dog_preds), 4) if dog_preds else 0,
                'precision': round(precision_dog, 4),
                'recall': round(recall_dog, 4)
            },
            'latency': {
                'avg_ms': round(avg_latency, 2),
                'p95_ms': round(p95_latency, 2)
            },
            'avg_confidence': round(avg_confidence, 4)
        }

        return metrics

    def check_api_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_api_stats(self) -> Dict:
        """Get stats from the API."""
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=5)
            return response.json()
        except Exception:
            return {}

    def save_predictions(self, filename: str = None):
        """Save predictions to file."""
        if filename is None:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.predictions_log, f, indent=2)

        print(f"Predictions saved to {filepath}")
        return filepath

    def save_metrics(self, metrics: Dict, filename: str = None):
        """Save metrics to file."""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {filepath}")
        return filepath

    def print_report(self, metrics: Dict):
        """Print a formatted performance report."""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE REPORT")
        print("="*60)
        print(f"Timestamp: {metrics['timestamp']}")
        print(f"Total Predictions: {metrics['total_predictions']}")
        print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
        print(f"Average Confidence: {metrics['avg_confidence']:.2%}")
        print(f"\nLatency:")
        print(f"  Average: {metrics['latency']['avg_ms']:.2f} ms")
        print(f"  P95: {metrics['latency']['p95_ms']:.2f} ms")
        print(f"\nCat Classification:")
        print(f"  Total: {metrics['cat']['total']}")
        print(f"  Accuracy: {metrics['cat']['accuracy']:.2%}")
        print(f"  Precision: {metrics['cat']['precision']:.2%}")
        print(f"  Recall: {metrics['cat']['recall']:.2%}")
        print(f"\nDog Classification:")
        print(f"  Total: {metrics['dog']['total']}")
        print(f"  Accuracy: {metrics['dog']['accuracy']:.2%}")
        print(f"  Precision: {metrics['dog']['precision']:.2%}")
        print(f"  Recall: {metrics['dog']['recall']:.2%}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Track model performance')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000',
                        help='API base URL')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of predictions to collect')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')
    parser.add_argument('--check-api', action='store_true',
                        help='Check API health before collecting')

    args = parser.parse_args()

    tracker = PerformanceTracker(api_url=args.api_url, log_dir=args.log_dir)

    # Check API health
    if args.check_api:
        print(f"Checking API health at {args.api_url}...")
        if tracker.check_api_health():
            print("✓ API is healthy")
            stats = tracker.get_api_stats()
            if stats:
                print(f"  Total requests: {stats.get('total_requests', 'N/A')}")
        else:
            print("✗ API is not available")

    # Collect predictions
    predictions = tracker.collect_predictions(args.samples)

    # Calculate metrics
    metrics = tracker.calculate_metrics(predictions)

    # Print report
    tracker.print_report(metrics)

    # Save results
    tracker.save_predictions()
    tracker.save_metrics(metrics)

    print("\nPerformance tracking complete!")


if __name__ == '__main__':
    main()
