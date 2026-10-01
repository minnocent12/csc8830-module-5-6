"""Image presentation around native ``st.image``: card, side-by-side comparison, gallery.

Presentation only. No component converts color spaces, rescales values, or changes dtype.
The caller passes a display-ready image and states its channel order with ``channels``
("RGB" or "BGR", exactly as ``st.image`` defines it); arrays are handed to ``st.image``
unchanged, together with ``clamp`` and ``output_format`` when the caller sets them.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import streamlit as st

from ._grid import grid_rows
from ._text import markdown_text

DEFAULT_GALLERY_COLUMNS = 3
MAX_GALLERY_COLUMNS = 4


@dataclass(frozen=True)
class ImageItem:
    """One image plus its labels and the native ``st.image`` options it needs."""

    image: Any
    title: str | None = None
    caption: str | None = None
    note: str | None = None
    channels: str = "RGB"
    clamp: bool = False
    output_format: str = "auto"


def image_card(
    image: Any,
    *,
    title: str | None = None,
    caption: str | None = None,
    note: str | None = None,
    channels: str = "RGB",
    clamp: bool = False,
    output_format: str = "auto",
    width: int | str = "stretch",
) -> None:
    """Bordered image with an optional bold title above and a note below the native caption."""
    with st.container(border=True):
        if title:
            st.markdown(f"**{markdown_text(title)}**")
        st.image(
            image,
            caption=caption,
            width=width,
            channels=channels,
            clamp=clamp,
            output_format=output_format,
        )
        if note:
            st.caption(note)


def _render(item: ImageItem, width: int | str) -> None:
    image_card(
        item.image,
        title=item.title,
        caption=item.caption,
        note=item.note,
        channels=item.channels,
        clamp=item.clamp,
        output_format=item.output_format,
        width=width,
    )


def image_comparison(left: ImageItem, right: ImageItem, *, width: int | str = "stretch") -> None:
    """Two images side by side, for example spatial and Fourier results.

    Uses two equal ``st.columns``; Streamlit stacks columns on narrow screens, and a later
    responsive pass can adjust the breakpoint without changing this API.
    """
    left_column, right_column = st.columns(2)
    with left_column:
        _render(left, width)
    with right_column:
        _render(right, width)


def image_gallery(
    items: Sequence[ImageItem],
    *,
    max_columns: int = DEFAULT_GALLERY_COLUMNS,
    width: int | str = "stretch",
) -> None:
    """Any number of images in balanced rows of at most ``max_columns`` (three by default)."""
    if not 1 <= max_columns <= MAX_GALLERY_COLUMNS:
        raise ValueError(f"max_columns must be between 1 and {MAX_GALLERY_COLUMNS}")
    columns, rows = grid_rows(list(items), max_columns)
    for row in rows:
        for column, item in zip(st.columns(columns), row):
            with column:
                _render(item, width)
