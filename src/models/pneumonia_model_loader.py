"""Loader for pneumonia DenseNet Keras model."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import tensorflow as tf

from config.paths import get_task_manifest, resolve_path
from src.models import mlflow_registry
from src.utils.file_utils import ensure_file_exists
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def load_pneumonia_artifacts() -> dict[str, Any]:
    """Load Keras model for pneumonia detection from MLflow or local file."""
    task = get_task_manifest("pneumonia")

    if mlflow_registry.enabled():
        try:
            model = mlflow_registry.load_keras(mlflow_registry.PNEUMONIA_MODEL)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed loading pneumonia model from MLflow.")
            raise RuntimeError(
                "Failed to load pneumonia model from MLflow.") from exc
        LOGGER.info("Loaded pneumonia model from MLflow registry.")
        return {"model": model, "task_manifest": task}

    model_path = ensure_file_exists(resolve_path(
        task["model_path"]), "pneumonia keras model")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed loading pneumonia model artifact.")
        raise RuntimeError("Failed to load pneumonia model artifact.") from exc

    LOGGER.info("Loaded pneumonia model successfully.")
    return {
        "model": model,
        "task_manifest": task,
    }
