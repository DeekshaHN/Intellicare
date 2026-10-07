"""Helpers to load models from the MLflow Model Registry.

MLflow is used only when the environment variable MLFLOW_TRACKING_URI is set.
If it is not set, the app keeps loading models from the local artifacts/ folder.
"""

from __future__ import annotations

import os

# Names used in register_models.py
HEART_MODEL = "heart-pipeline"
DIABETES_MODEL = "diabetes-stack"
PNEUMONIA_MODEL = "pneumonia-densenet121"
ALIAS = os.environ.get("MLFLOW_MODEL_ALIAS", "production")


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _setup():
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    return mlflow


def load_sklearn(name: str, alias: str = ALIAS):
    mlflow = _setup()
    import mlflow.sklearn

    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


def load_keras(name: str, alias: str = ALIAS):
    mlflow = _setup()
    import mlflow.keras

    return mlflow.keras.load_model(f"models:/{name}@{alias}")


def download_run_artifact(name: str, artifact_path: str, alias: str = ALIAS) -> str:
    """Download an extra file logged in the same run as the model version."""
    mlflow = _setup()
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    return mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path=artifact_path
    )
