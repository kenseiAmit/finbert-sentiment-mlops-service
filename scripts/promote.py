"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""

import os

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    """
    Return the run_id and f1_weighted of the most recent evaluation run.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    f1_weighted = latest_run.data.metrics.get("f1_weighted")
    if f1_weighted is None:
        raise ValueError(f"Run {run_id} does not contain 'f1_weighted' metric.")

    return run_id, f1_weighted


def promote(client: MlflowClient, model_name: str):
    """
    Fetch all registered versions of the model, identify the latest one,
    and assign it the "production" alias using the MLflow client.
    """
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if not model_versions:
        raise ValueError(f"No model versions found for model '{model_name}'.")

    latest_version = max(model_versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version.version,
    )
    print(
        f"Model '{model_name}' version {latest_version.version} successfully assigned 'production' alias."
    )


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
    model_name = os.getenv("MODEL_NAME", "finbert")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    params = load_params()
    f1_threshold = params["f1_threshold"]

    run_id, f1_weighted = get_latest_f1(client, experiment_name)
    print(f"Latest evaluation run ({run_id}) f1_weighted: {f1_weighted:.4f}")
    print(f"Promotion threshold: {f1_threshold:.4f}")

    if f1_weighted >= f1_threshold:
        promote(client, model_name)
    else:
        print(
            f"Model does not meet the threshold: f1_weighted ({f1_weighted:.4f}) < threshold ({f1_threshold:.4f}). Promotion skipped."
        )


if __name__ == "__main__":
    main()
