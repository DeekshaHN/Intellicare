"""FastAPI backend: loads models from MLflow and serves predictions + explanations.

Run locally:
    set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on a server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile  # noqa: E402
from PIL import Image  # noqa: E402

from src.explainability.explanation_rules import (  # noqa: E402
    generate_image_explanation,
    generate_tabular_explanation,
)
from src.explainability.feature_name_mapping import humanize_feature  # noqa: E402
from src.explainability.shap_runtime import generate_local_shap_figure  # noqa: E402
from src.models.diabetes_model_loader import load_diabetes_artifacts  # noqa: E402
from src.models.heart_model_loader import load_heart_artifacts  # noqa: E402
from src.models.pneumonia_model_loader import load_pneumonia_artifacts  # noqa: E402
from src.pipelines.diabetes_pipeline import predict_diabetes  # noqa: E402
from src.pipelines.heart_pipeline import predict_heart  # noqa: E402
from src.pipelines.pneumonia_pipeline import predict_pneumonia  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.validators import validate_uploaded_filename  # noqa: E402

LOGGER = get_logger(__name__)

MODEL_STATUS: dict[str, str] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load all three models once at startup (from MLflow when configured)."""
    loaders = {
        "diabetes": load_diabetes_artifacts,
        "heart": load_heart_artifacts,
        "pneumonia": load_pneumonia_artifacts,
    }
    for name, loader in loaders.items():
        try:
            loader()
            MODEL_STATUS[name] = "loaded"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to load %s model at startup.", name)
            MODEL_STATUS[name] = f"error: {exc}"
    yield


app = FastAPI(title="XAI Medical Diagnosis API", version="1.0.0", lifespan=lifespan)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If API_KEY is set in the environment, require it in the X-API-Key header."""
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _fig_to_b64(fig: Any) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _array_to_b64(array: Any) -> str:
    image = Image.fromarray(np.asarray(array).astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _tabular_response(task_name: str, task_key: str, result: Any, model: Any, model_input: Any) -> dict[str, Any]:
    summary = generate_tabular_explanation(
        task_name=task_name,
        predicted_label=result.label,
        predicted_probability=result.probability,
        confidence_band=result.confidence_band,
        positive_contributors=result.positive_contributors,
        negative_contributors=result.negative_contributors,
    )

    shap_image: str | None = None
    shap_error: str | None = None
    try:
        shap_result = generate_local_shap_figure(
            task_key=task_key, model=model, model_input=model_input)
        if shap_result.success and shap_result.figure is not None:
            shap_image = _fig_to_b64(shap_result.figure)
        else:
            shap_error = shap_result.error
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("SHAP generation failed for %s.", task_key)
        shap_error = str(exc)

    return {
        "label": result.label,
        "probability": result.probability,
        "confidence_band": result.confidence_band,
        "positive_factors": [humanize_feature(f) for f in result.positive_contributors],
        "negative_factors": [humanize_feature(f) for f in result.negative_contributors],
        "summary": summary,
        "shap_image_b64": shap_image,
        "shap_error": shap_error,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    ok = all(status == "loaded" for status in MODEL_STATUS.values()) and bool(MODEL_STATUS)
    return {"status": "ok" if ok else "degraded", "models": MODEL_STATUS}


@app.post("/predict/diabetes", dependencies=[Depends(require_api_key)])
def predict_diabetes_endpoint(payload: dict[str, float]) -> dict[str, Any]:
    try:
        result = predict_diabetes(payload)
        artifacts = load_diabetes_artifacts()
        return _tabular_response(
            "Diabetes Risk Prediction", "diabetes", result,
            artifacts["model"], result.scaled_features)
    except KeyError as exc:
        raise HTTPException(
            status_code=422, detail=f"Missing or unknown input field: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Diabetes prediction failed.")
        raise HTTPException(status_code=500, detail=f"Diabetes prediction failed: {exc}") from exc


@app.post("/predict/heart", dependencies=[Depends(require_api_key)])
def predict_heart_endpoint(payload: dict[str, float]) -> dict[str, Any]:
    try:
        result = predict_heart(payload)
        artifacts = load_heart_artifacts()
        return _tabular_response(
            "Heart Disease Prediction", "heart", result,
            artifacts["pipeline"], result.model_input)
    except KeyError as exc:
        raise HTTPException(
            status_code=422, detail=f"Missing or unknown input field: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Heart prediction failed.")
        raise HTTPException(status_code=500, detail=f"Heart prediction failed: {exc}") from exc


@app.post("/predict/pneumonia", dependencies=[Depends(require_api_key)])
def predict_pneumonia_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    valid, message = validate_uploaded_filename(file.filename or "")
    if not valid:
        raise HTTPException(status_code=422, detail=message)

    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        result = predict_pneumonia(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Pneumonia prediction failed.")
        raise HTTPException(status_code=500, detail=f"Pneumonia prediction failed: {exc}") from exc

    summary = generate_image_explanation(
        predicted_label=result.label,
        predicted_probability=result.probability,
        confidence_band=result.confidence_band,
    )
    return {
        "label": result.label,
        "probability": result.probability,
        "confidence_band": result.confidence_band,
        "summary": summary,
        "gradcam_success": bool(result.gradcam_success),
        "original_b64": _array_to_b64(result.original_image),
        "overlay_b64": _array_to_b64(result.gradcam_overlay) if result.gradcam_overlay is not None else None,
        "heatmap_b64": _array_to_b64(result.gradcam_heatmap) if result.gradcam_heatmap is not None else None,
    }
