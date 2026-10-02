"""Small Streamlit UI helpers for the foundation phase."""
from __future__ import annotations

import streamlit as st

VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "m4v"]
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def pending_experiment_banner(detail: str | None = None) -> None:
    """Display an honest notice for work that needs later implementation or user data."""
    message = (
        "**PENDING USER EXPERIMENT.** "
        "No empirical result is available for this section yet."
    )
    st.warning(message if detail is None else f"{message}\n\n{detail}")


def foundation_page(title: str, scope: str) -> None:
    """Render a phase-0 placeholder without pretending that algorithms exist."""
    st.header(title)
    st.write(scope)
    pending_experiment_banner(
        "This page is available, but its computer-vision processing is not implemented "
        "yet."
    )
