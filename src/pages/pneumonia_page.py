"""Pneumonia detection page."""

from __future__ import annotations

import streamlit as st

from config.paths import get_task_manifest, resolve_path
from src.components.disclaimer import render_disclaimer
from src.components.figure_display import display_hero_image, display_image_group, display_optional_image
from src import api_client
from src.api_client import ApiError, b64_to_bytes
from src.components.header import render_page_header
from src.components.metric_cards import render_prediction_metrics
from src.utils.image_utils import decode_uploaded_image
from src.utils.logger import get_logger
from src.utils.session_state import save_prediction
from src.utils.validators import validate_uploaded_filename

LOGGER = get_logger(__name__)


def render_pneumonia_page() -> None:
    """Render pneumonia image inference and Grad-CAM results."""
    render_page_header("Pneumonia Detection",
                       "Chest X-ray inference with Grad-CAM attention visualization")
    render_disclaimer()

    uploaded_file = st.file_uploader(
        "Upload chest X-ray image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

    uploaded_bytes: bytes | None = None
    upload_is_valid = False

    if uploaded_file is not None:
        valid_name, name_message = validate_uploaded_filename(
            uploaded_file.name)
        if not valid_name:
            st.error(name_message)
        else:
            try:
                uploaded_bytes = uploaded_file.getvalue()
                preview_image = decode_uploaded_image(uploaded_bytes)
                upload_is_valid = True
                st.image(preview_image, caption="Uploaded X-ray",
                         use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to preview uploaded pneumonia image.")
                st.error(
                    "Uploaded file could not be decoded as an image. Please upload a valid PNG/JPG/BMP/TIFF file.")

    if st.button("Predict Pneumonia Pattern", type="primary"):
        if not upload_is_valid or uploaded_bytes is None:
            if uploaded_file is None:
                st.error("No file uploaded.")
            else:
                st.error(
                    "Please upload a valid image before running prediction.")
            return

        try:
            with st.spinner("Running prediction on the API..."):
                result = api_client.predict_pneumonia(
                    uploaded_bytes, uploaded_file.name)
            render_prediction_metrics(
                result["label"], result["probability"], result["confidence_band"])

            st.subheader("Rule-Based Clinical Summary")
            st.info(result["summary"])
            st.caption(
                "This heatmap shows image regions that influenced the model output and supports interpretation of model behavior; it is not a final diagnosis."
            )
            save_prediction(
                task="pneumonia",
                label=result["label"],
                probability=result["probability"],
                confidence_band=result["confidence_band"],
                summary=result["summary"],
            )

            st.subheader("Grad-CAM Visualization")
            if result["gradcam_success"] and result.get("overlay_b64"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(b64_to_bytes(result["original_b64"]),
                             caption="Original X-ray", use_container_width=True)
                with col2:
                    st.image(b64_to_bytes(result["overlay_b64"]),
                             caption="Grad-CAM Overlay", use_container_width=True)
                with col3:
                    if result.get("heatmap_b64"):
                        st.image(b64_to_bytes(result["heatmap_b64"]),
                                 caption="Grad-CAM Heatmap", use_container_width=True)
                    else:
                        st.warning(
                            "Heatmap view is unavailable for this prediction.")
            else:
                st.warning(
                    "Prediction completed, but the Grad-CAM visualization could not be generated. Showing a static fallback figure when available."
                )
                fallback_paths = [
                    resolve_path(p)
                    for p in get_task_manifest("pneumonia")["static_figures"]["gradcam_fallback"]
                ]
                fallback = next((p for p in fallback_paths if p.is_file()), None)
                if fallback is not None:
                    display_optional_image(fallback, caption=fallback.name)
        except ApiError as exc:
            LOGGER.error("Pneumonia prediction failed: %s", exc)
            st.error(f"Pneumonia prediction failed: {exc}")

    task_manifest = get_task_manifest("pneumonia")
    overview_paths = [
        resolve_path(path_str) for path_str in task_manifest["static_figures"]["overview"]
    ]

    learning_curve_path = next(
        (
            path
            for path in overview_paths
            if "learning_curve" in path.name.lower() or "learning_curves" in path.name.lower()
        ),
        None,
    )
    remaining_overview_paths = [
        path for path in overview_paths if path != learning_curve_path
    ]

    if learning_curve_path is not None:
        st.subheader("Pneumonia Learning Curve")
        display_hero_image(
            learning_curve_path,
            caption=learning_curve_path.name,
        )

    st.subheader("Static Pneumonia Figures")
    display_image_group(
        remaining_overview_paths,
        columns=2,
        image_width=560,
        use_container_width=True,
    )

    with st.expander("Show static Grad-CAM fallback gallery"):
        fallback_paths = [
            resolve_path(path_str)
            for path_str in task_manifest["static_figures"]["gradcam_fallback"]
        ]
        display_image_group(
            fallback_paths,
            columns=2,
            image_width=520,
            use_container_width=True,
        )

    render_disclaimer()
