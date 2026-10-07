"""Loader for diabetes model and preprocessing artifacts."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib

from config.paths import get_task_manifest, resolve_path
from src.models import mlflow_registry
from src.utils.file_utils import ensure_file_exists
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def load_diabetes_artifacts() -> dict[str, Any]:
    """Load diabetes model, imputer, and scaler from MLflow or local files."""
    task = get_task_manifest("diabetes")

    if mlflow_registry.enabled():
        try:
            name = mlflow_registry.DIABETES_MODEL
            model = mlflow_registry.load_sklearn(name)
            imputer = joblib.load(mlflow_registry.download_run_artifact(
                name, "preprocessors/diab_knn_imp.pkl"))
            scaler = joblib.load(mlflow_registry.download_run_artifact(
                name, "preprocessors/diab_scaler.pkl"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed loading diabetes artifacts from MLflow.")
            raise RuntimeError(
                "Failed to load diabetes artifacts from MLflow.") from exc
        LOGGER.info("Loaded diabetes artifacts from MLflow registry.")
        return {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "task_manifest": task,
        }

    model_path = ensure_file_exists(
        resolve_path(task["model_path"]), "diabetes model")
    imputer_path = ensure_file_exists(
        resolve_path(task["preprocessors"]["imputer_path"]),
        "diabetes KNN imputer",
    )
    scaler_path = ensure_file_exists(
        resolve_path(task["preprocessors"]["scaler_path"]),
        "diabetes scaler",
    )

    try:
        model = joblib.load(model_path)
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
    except ModuleNotFoundError as exc:
        missing_module = exc.name or str(exc)
        LOGGER.exception(
            "Failed loading diabetes artifacts due to missing dependency module: %s",
            missing_module,
        )
        raise RuntimeError(
            "Failed to load diabetes model artifacts. "
            f"Missing dependency: {missing_module}. "
            "Install it in your active environment and retry."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed loading diabetes artifacts.")
        raise RuntimeError("Failed to load diabetes model artifacts.") from exc

    LOGGER.info("Loaded diabetes artifacts successfully.")
    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "task_manifest": task,
    }
