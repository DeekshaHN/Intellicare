"""Thin HTTP client the Streamlit pages use to call the FastAPI backend."""

from __future__ import annotations

import base64
import os
from typing import Any

import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
TIMEOUT_SECONDS = float(os.environ.get("API_TIMEOUT", "180"))


class ApiError(RuntimeError):
    """Raised when the backend is unreachable or returns an error."""


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _handle(response: requests.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(str(detail))
    return response.json()


def _call(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(
            method, f"{API_URL}{path}", headers=_headers(), timeout=TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the prediction API at {API_URL}: {exc}") from exc
    return _handle(response)


def health() -> dict[str, Any]:
    return _call("GET", "/health")


def predict_diabetes(payload: dict[str, float]) -> dict[str, Any]:
    return _call("POST", "/predict/diabetes", json=payload)


def predict_heart(payload: dict[str, float]) -> dict[str, Any]:
    return _call("POST", "/predict/heart", json=payload)


def predict_pneumonia(file_bytes: bytes, filename: str) -> dict[str, Any]:
    return _call("POST", "/predict/pneumonia", files={"file": (filename, file_bytes)})


def b64_to_bytes(value: str) -> bytes:
    return base64.b64decode(value)
