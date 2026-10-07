"""Streamlit rendering for results returned by the FastAPI backend."""

from __future__ import annotations

import streamlit as st

from src.api_client import b64_to_bytes
from src.components.figure_display import display_optional_image
from src.components.metric_cards import render_prediction_metrics
from src.explainability.fallback_loader import get_tabular_fallback_image
from src.utils.session_state import save_prediction


def render_tabular_result(task_key: str, result: dict) -> None:
    """Render metrics, text explanation and SHAP graph for diabetes/heart."""
    render_prediction_metrics(
        result["label"], result["probability"], result["confidence_band"])

    st.subheader("Local Explanation")
    positive = ", ".join(result["positive_factors"])
    negative = ", ".join(result["negative_factors"])
    st.write(f"Key risk-increasing factors: {positive or 'Not identified'}")
    st.write(f"Potential risk-reducing factors: {negative or 'Not identified'}")

    st.subheader("Rule-Based Clinical Summary")
    st.info(result["summary"])

    st.subheader("Feature Contribution Graph")
    st.caption(
        "This graph shows which features influenced the model prediction. "
        "It supports interpretation of model behavior and is not a final diagnosis.")
    if result.get("shap_image_b64"):
        st.image(b64_to_bytes(result["shap_image_b64"]), use_container_width=True)
    else:
        fallback = get_tabular_fallback_image(task_key)
        if fallback is not None:
            st.warning(
                "A live explanation graph could not be generated. Showing a static fallback figure.")
            display_optional_image(
                fallback, caption=fallback.name, use_container_width=True, width=None)
        else:
            st.warning(
                "Prediction completed, but the explanation graph is unavailable for this input.")

    save_prediction(
        task=task_key, label=result["label"], probability=result["probability"],
        confidence_band=result["confidence_band"], summary=result["summary"])
