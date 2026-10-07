"""Diabetes prediction page."""

from __future__ import annotations

import streamlit as st

from config.paths import get_task_manifest, resolve_path
from src.components.disclaimer import render_disclaimer
from src.components.figure_display import display_image_group
from src.api_client import ApiError
from src import api_client
from src.components.api_results import render_tabular_result
from src.components.header import render_page_header
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _render_diabetes_form() -> dict[str, float]:
    """Render diabetes input controls and return payload."""
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
        glucose = st.number_input(
            "Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input(
            "BloodPressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        skin_thickness = st.number_input(
            "SkinThickness", min_value=0.0, max_value=120.0, value=20.0, step=1.0)
    with col2:
        insulin = st.number_input(
            "Insulin", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0,
                              max_value=80.0, value=28.0, step=0.1)
        dpf = st.number_input(
            "DiabetesPedigreeFunction",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            format="%.3f",
        )
        age = st.number_input("Age", min_value=20.0,
                              max_value=100.0, value=45.0, step=1.0)

    return {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }


def render_diabetes_page() -> None:
    """Render diabetes prediction workflow page."""
    render_page_header("Diabetes Prediction",
                       "Tabular model inference with rule-based explanation")
    render_disclaimer()

    st.subheader("Patient Inputs")
    payload = _render_diabetes_form()

    if st.button("Predict Diabetes Risk", type="primary"):
        try:
            with st.spinner("Running prediction on the API..."):
                result = api_client.predict_diabetes(payload)
            render_tabular_result("diabetes", result)
        except ApiError as exc:
            LOGGER.error("Diabetes prediction failed: %s", exc)
            st.error(f"Diabetes prediction failed: {exc}")

    task_manifest = get_task_manifest("diabetes")
    st.subheader("Static Explainability Figures")
    required_figures = [
        resolve_path("static_outputs/diabetes/shap_bar_Diabetes.png"),
        resolve_path("static_outputs/diabetes/shap_waterfall_Diabetes_s1.png"),
        resolve_path(
            "static_outputs/diabetes/lime_Diabetes_HighRisk_Positive.png"),
    ]
    display_image_group(
        required_figures,
        columns=2,
        image_width=560,
        use_container_width=True,
    )

    extra_xai = [resolve_path(path)
                 for path in task_manifest["static_figures"]["xai"]]
    with st.expander("Show additional diabetes XAI figures"):
        display_image_group(
            extra_xai,
            columns=2,
            image_width=520,
            use_container_width=True,
        )

    render_disclaimer()
