"""Loader for heart disease pipeline artifact."""

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
def load_heart_artifacts() -> dict[str, Any]:
    """Load heart sklearn pipeline from MLflow (if configured) or local file."""
    task = get_task_manifest("heart")

    if mlflow_registry.enabled():
        try:
            pipeline = mlflow_registry.load_sklearn(mlflow_registry.HEART_MODEL)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed loading heart pipeline from MLflow.")
            raise RuntimeError("Failed to load heart model from MLflow.") from exc
        LOGGER.info("Loaded heart pipeline from MLflow registry.")
        return {"pipeline": pipeline, "task_manifest": task}

    model_path = ensure_file_exists(resolve_path(
        task["model_path"]), "heart pipeline model")

    try:
        pipeline = joblib.load(model_path)
    except AttributeError as exc:
        message = str(exc)
        if "_RemainderColsList" in message:
            LOGGER.exception(
                "Failed loading heart pipeline due to scikit-learn version mismatch.")
            raise RuntimeError(
                "Failed to load heart model artifact. The saved pipeline was built with a different "
                "scikit-learn version. Install scikit-learn==1.6.1 and retry."
            ) from exc
        LOGGER.exception("Failed loading heart pipeline artifact.")
        raise RuntimeError("Failed to load heart model artifact.") from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed loading heart pipeline artifact.")
        raise RuntimeError("Failed to load heart model artifact.") from exc

    LOGGER.info("Loaded heart pipeline successfully.")
    return {
        "pipeline": pipeline,
        "task_manifest": task,
    }
