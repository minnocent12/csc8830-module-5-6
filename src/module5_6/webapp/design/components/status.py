"""Status vocabulary: compact chips, native alert banners, and empty, pending, success states.

Chips are the only status element rendered as custom HTML. Each chip always shows its text
label, so color is never the only indicator, and the label is entity-escaped. Banners and
the pending and success states are native ``st.info``, ``st.success``, ``st.warning`` and
``st.error`` calls, so tests and assistive technology see real alerts. All wording is supplied
by the caller; nothing here rewrites or standardizes module text.
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import streamlit as st

from ..styles import CSS_PREFIX
from ._text import html_text, markdown_text


class StatusKind(str, Enum):
    """Semantic chip styles. Each maps to a token color pair checked for AA contrast."""

    NEUTRAL = "neutral"  # text_body on surface_secondary
    INFO = "info"  # info on info_soft
    SUCCESS = "success"  # success on success_soft
    WARNING = "warning"  # warning on warning_soft
    ERROR = "error"  # error on error_soft
    BRAND = "brand"  # brand_orange_strong on brand_orange_soft


BANNER_KINDS = ("info", "success", "warning", "error")
ChipSpec = tuple[str, StatusKind | str]


def _kind(kind: StatusKind | str) -> StatusKind:
    try:
        return StatusKind(kind)
    except ValueError:
        choices = ", ".join(k.value for k in StatusKind)
        raise ValueError(f"unknown status kind {kind!r}; choose from {choices}") from None


def chip_html(label: str, kind: StatusKind | str = StatusKind.NEUTRAL) -> str:
    """One escaped chip element. The label is required and is always visible text."""
    if not str(label).strip():
        raise ValueError("a status chip needs a visible text label")
    k = _kind(kind).value
    return f'<span class="{CSS_PREFIX}-chip {CSS_PREFIX}-chip--{k}">{html_text(label)}</span>'


def chips_html(chips: Iterable[ChipSpec]) -> str:
    """A wrapping row of escaped chips."""
    inner = "".join(chip_html(label, kind) for label, kind in chips)
    return f'<div class="{CSS_PREFIX}-chip-row">{inner}</div>'


def status_chip(label: str, kind: StatusKind | str = StatusKind.NEUTRAL) -> None:
    """Render one status chip, for example ``status_chip("Passed", "success")``."""
    st.html(chips_html([(label, kind)]))


def status_chips(chips: Iterable[ChipSpec]) -> None:
    """Render several chips on one wrapping row: ``[("Complete", "success"), ...]``."""
    chips = list(chips)
    if chips:
        st.html(chips_html(chips))


def status_banner(
    kind: str, message: str, *, title: str | None = None, icon: str | None = None
) -> None:
    """Native alert. ``kind`` is one of info, success, warning, error.

    Use banners when the reader needs to act or must not miss the message; prefer chips for
    routine state. ``message`` is Markdown; ``title`` is shown in bold and escaped.
    """
    renderers = {"info": st.info, "success": st.success, "warning": st.warning, "error": st.error}
    if kind not in renderers:
        raise ValueError(f"unknown banner kind {kind!r}; choose from {', '.join(BANNER_KINDS)}")
    body = f"**{markdown_text(title)}**\n\n{message}" if title else message
    renderers[kind](body, icon=icon)


def empty_state(title: str, message: str | None = None) -> None:
    """Quiet bordered placeholder for 'nothing to show yet', without an alert color."""
    with st.container(border=True):
        st.markdown(f"**{markdown_text(title)}**")
        if message:
            st.caption(message)


def pending_state(message: str, *, icon: str | None = None) -> None:
    """Pending work or missing experimental data, as a native warning. Wording is the caller's."""
    st.warning(message, icon=icon)


def success_state(message: str, *, icon: str | None = None) -> None:
    """A completed or validated result, as a native success alert. Wording is the caller's."""
    st.success(message, icon=icon)
