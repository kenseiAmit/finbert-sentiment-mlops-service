"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml
from app.utils import load_classifier
from deepchecks.nlp import TextData

try:
    from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
except ImportError:
    from deepchecks.nlp.checks import PredictionDrift
    from deepchecks.nlp.checks import TextPropertyDrift as PropertyDrift
from dotenv import load_dotenv

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32, truncation=True, max_length=512)
    return [r["label"] for r in results]


def extract_drift_value(result_val, key: str | None = None) -> float:
    """Helper to extract float drift score from deepchecks check result."""
    if isinstance(result_val, dict):
        if key and key in result_val:
            val = result_val[key]
            if isinstance(val, dict):
                score = val.get("Drift score", val.get("drift_score"))
                if score is not None:
                    return float(score)
                for v in val.values():
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
        if "Drift score" in result_val:
            return float(result_val["Drift score"])
        if "drift_score" in result_val:
            return float(result_val["drift_score"])
        for v in result_val.values():
            if isinstance(v, dict):
                score = v.get("Drift score", v.get("drift_score"))
                if score is not None:
                    return float(score)
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    elif hasattr(result_val, "loc") and key in result_val.index:
        val = result_val.loc[key]
        if isinstance(val, dict):
            return float(val.get("Drift score", val.get("drift_score", 0.0)))
        return float(val)
    elif hasattr(result_val, "item"):
        return float(result_val.item())

    try:
        return float(result_val)
    except (TypeError, ValueError):
        return 0.0


def main():
    params = load_params()
    property_drift_threshold = float(params["property_drift_threshold"])
    prediction_drift_threshold = float(params["prediction_drift_threshold"])

    print("Loading production model...")
    classifier = load_classifier()

    stream_df = pd.read_csv("data/stream.csv")
    test_df = pd.read_csv("data/test.csv")

    stream_texts = stream_df["text"].dropna().tolist()
    test_texts = test_df["text"].dropna().tolist()

    print("Building TextData objects and calculating builtin properties...")
    test_dataset = TextData(test_texts, task_type="text_classification")
    test_dataset.calculate_builtin_properties()

    stream_dataset = TextData(stream_texts, task_type="text_classification")
    stream_dataset.calculate_builtin_properties()

    print("Running NLP Property Drift on 'Sentiment' property...")
    prop_check = PropertyDrift(properties=["Sentiment"])
    prop_result = prop_check.run(
        train_dataset=test_dataset, test_dataset=stream_dataset
    )
    sentiment_drift_score = extract_drift_value(prop_result.value, key="Sentiment")
    print(f"Sentiment Property Drift score: {sentiment_drift_score:.4f}")

    print("Running model predictions on test and stream datasets...")
    test_preds = run_predictions(classifier, test_texts)
    stream_preds = run_predictions(classifier, stream_texts)

    print("Running Prediction Drift check...")
    pred_check = PredictionDrift()
    pred_result = pred_check.run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset,
        train_predictions=test_preds,
        test_predictions=stream_preds,
    )
    prediction_drift_score = extract_drift_value(pred_result.value)
    print(f"Prediction Drift score: {prediction_drift_score:.4f}")

    failed = False
    if sentiment_drift_score > property_drift_threshold:
        print(
            f"[ERROR] Property drift score ({sentiment_drift_score:.4f}) exceeds threshold ({property_drift_threshold:.4f})"
        )
        failed = True
    else:
        print(
            f"[PASS] Property drift score ({sentiment_drift_score:.4f}) <= threshold ({property_drift_threshold:.4f})"
        )

    if prediction_drift_score > prediction_drift_threshold:
        print(
            f"[ERROR] Prediction drift score ({prediction_drift_score:.4f}) exceeds threshold ({prediction_drift_threshold:.4f})"
        )
        failed = True
    else:
        print(
            f"[PASS] Prediction drift score ({prediction_drift_score:.4f}) <= threshold ({prediction_drift_threshold:.4f})"
        )

    if failed:
        sys.exit(1)

    print("All Deepchecks drift checks passed successfully.")


if __name__ == "__main__":
    main()
