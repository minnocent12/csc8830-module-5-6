"""Upload panel around native ``st.file_uploader``.

The uploader stays the native widget, created with exactly the caller's label, types, key,
and options, so pages keep their widget identity and AppTest's ``file_uploader`` finds it.
The panel only adds optional explanation text and the bundled-sample notice. Whether a
bundled sample is actually used, and how files are decoded, stays in the page.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st


def supported_formats_text(types: Sequence[str]) -> str:
    """``["jpg", "png"]`` becomes ``"Supported formats: JPG, PNG"``, keeping caller order."""
    seen: list[str] = []
    for item in types:
        name = item.lstrip(".").upper()
        if name and name not in seen:
            seen.append(name)
    return "Supported formats: " + ", ".join(seen)


def upload_panel(
    label: str,
    *,
    type: Sequence[str] | None,
    key: str | None = None,
    accept_multiple_files: bool = False,
    help: str | None = None,
    disabled: bool = False,
    label_visibility: str = "visible",
    explanation: str | None = None,
    sample_notice: str | None = None,
    show_formats: bool = False,
) -> Any:
    """Render the native uploader and return its value unchanged.

    ``sample_notice`` is shown as a native ``st.info`` only when nothing is uploaded, matching
    how pages already announce bundled real samples. ``show_formats`` adds a caption listing
    the accepted types; it is off by default because the native uploader lists them too.
    """
    if explanation:
        st.caption(explanation)
    value = st.file_uploader(
        label,
        type=type,  # passed through untouched: widget identity depends on the arguments
        accept_multiple_files=accept_multiple_files,
        key=key,
        help=help,
        disabled=disabled,
        label_visibility=label_visibility,
    )
    if show_formats and type:
        st.caption(supported_formats_text(type))
    if sample_notice and not value:
        st.info(sample_notice)
    return value
