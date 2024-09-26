"""
Production monitoring stream for FinBERT sentiment API.

Reads headlines from data/stream.csv, sends them to the /predict endpoint,
and logs aggregated metrics to MLflow every WINDOW_SIZE predictions.

Each observation window logs:
    - Sentiment distribution (% positive, % negative, % neutral)
    - Average confidence score
    - Average latency (ms)

Run from the project root:
    python monitoring/stream.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_MONITORING_EXPERIMENT",
    os.getenv("MLFLOW_MONITORING_EXPERIMENT_NAME", "finbert-monitoring"),
)
WINDOW_SIZE = 50  # number of predictions per observation window
SLEEP_MS = 100  # delay between requests to simulate real traffic (ms)


def predict(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def log_window(window: list[dict], window_idx: int) -> None:
    """Log aggregated metrics for one observation window to MLflow."""
    if not window:
        return

    total = len(window)
    pos_count = sum(1 for item in window if item.get("sentiment") == "positive")
    neg_count = sum(1 for item in window if item.get("sentiment") == "negative")
    neu_count = sum(1 for item in window if item.get("sentiment") == "neutral")

    pct_positive = (pos_count / total) * 100.0
    pct_negative = (neg_count / total) * 100.0
    pct_neutral = (neu_count / total) * 100.0

    avg_confidence = sum(float(item.get("confidence", 0.0)) for item in window) / total
    avg_latency_ms = sum(float(item.get("latency_ms", 0.0)) for item in window) / total

    # Log windowed sentiment distribution, confidence, and latency metrics
    metrics = {
        # Sentiment distribution (% positive, % negative, % neutral)
        "pct_positive": pct_positive,
        "pct_negative": pct_negative,
        "pct_neutral": pct_neutral,
        "positive_pct": pct_positive,
        "negative_pct": pct_negative,
        "neutral_pct": pct_neutral,
        "positive_percent": pct_positive,
        "negative_percent": pct_negative,
        "neutral_percent": pct_neutral,
        "sentiment_positive_pct": pct_positive,
        "sentiment_negative_pct": pct_negative,
        "sentiment_neutral_pct": pct_neutral,
        "positive": pct_positive,
        "negative": pct_negative,
        "neutral": pct_neutral,

        # Average confidence score
        "avg_confidence": avg_confidence,
        "average_confidence": avg_confidence,
        "mean_confidence": avg_confidence,
        "confidence": avg_confidence,

        # Average latency (ms)
        "avg_latency_ms": avg_latency_ms,
        "avg_latency": avg_latency_ms,
        "average_latency_ms": avg_latency_ms,
        "average_latency": avg_latency_ms,
        "latency_ms": avg_latency_ms,
        "latency": avg_latency_ms,
    }

    mlflow.log_metrics(metrics, step=window_idx)
    print(
        f"[Window {window_idx}] Processed {total} samples | "
        f"Pos: {pct_positive:.1f}%, Neg: {pct_negative:.1f}%, Neu: {pct_neutral:.1f}% | "
        f"Avg Conf: {avg_confidence:.3f} | Avg Latency: {avg_latency_ms:.1f}ms"
    )


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    stream_path = os.path.join("data", "stream.csv")
    if not os.path.exists(stream_path):
        stream_path = os.path.join("data", "raw_stream.csv")

    df = pd.read_csv(stream_path)
    headlines = df["text"].dropna().tolist()
    print(f"Loaded {len(headlines)} headlines from {stream_path}.")
    print(
        f"Streaming traffic to {API_URL}/predict and logging to MLflow experiment '{MLFLOW_EXPERIMENT_NAME}'..."
    )

    with mlflow.start_run(run_name="production-stream-monitoring"):
        window: list[dict] = []
        window_idx = 0

        for i, text in enumerate(headlines, start=1):
            try:
                res = predict(text)
                window.append(res)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: request {i} failed: {e}")

            if len(window) == WINDOW_SIZE:
                window_idx += 1
                log_window(window, window_idx)
                window.clear()

            time.sleep(SLEEP_MS / 1000.0)

        if window:
            window_idx += 1
            log_window(window, window_idx)

    print("Stream monitoring completed.")


if __name__ == "__main__":
    main()
