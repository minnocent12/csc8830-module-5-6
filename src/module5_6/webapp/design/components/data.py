"""Tables and downloads around native ``st.dataframe`` and ``st.download_button``.

Data is never converted to HTML: the native grid keeps sorting, copying, scrolling, and its
accessibility. Optional arguments are forwarded only when the caller sets them, so
Streamlit's own defaults apply otherwise.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st


def data_table(
    data: Any,
    *,
    column_config: Mapping[str, Any] | None = None,
    column_order: Sequence[str] | None = None,
    hide_index: bool | None = None,
    height: int | None = None,
    width: int | str = "stretch",
    key: str | None = None,
) -> Any:
    """Native dataframe; returns what ``st.dataframe`` returns."""
    options: dict[str, Any] = {"width": width}
    if column_config is not None:
        options["column_config"] = column_config
    if column_order is not None:
        options["column_order"] = column_order
    if hide_index is not None:
        options["hide_index"] = hide_index
    if height is not None:
        options["height"] = height
    if key is not None:
        options["key"] = key
    return st.dataframe(data, **options)


def download_action(
    label: str,
    data: Any,
    *,
    file_name: str,
    mime: str,
    key: str | None = None,
    help: str | None = None,
    primary: bool = False,
    disabled: bool = False,
) -> bool:
    """Native download button. The caller owns the bytes, file name, MIME type, and key."""
    return st.download_button(
        label,
        data,
        file_name=file_name,
        mime=mime,
        key=key,
        help=help,
        type="primary" if primary else "secondary",
        disabled=disabled,
    )
