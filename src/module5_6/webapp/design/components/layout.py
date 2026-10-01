"""Page-level structure: header, breadcrumbs, section header, footer.

Headings stay native (``st.header`` for the page title, ``st.subheader`` for sections) so the
document outline, anchors, and AppTest element lists are unchanged. Only the small eyebrow
and the breadcrumb trail are rendered with ``st.html``, and their text is entity-escaped.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import streamlit as st

from ..styles import CSS_PREFIX
from ._text import html_text
from .status import ChipSpec, status_chips

DEFAULT_FOOTER = ("CSc 8830 Computer Vision",)


def eyebrow_html(text: str) -> str:
    """Small uppercase context label above a page title, for example a module name."""
    return f'<p class="{CSS_PREFIX}-eyebrow">{html_text(text)}</p>'


def page_header(
    title: str,
    *,
    eyebrow: str | None = None,
    description: str | None = None,
    chips: Iterable[ChipSpec] | None = None,
    anchor: str | None = None,
) -> None:
    """Standard page opening: optional eyebrow, native page title, chips, description.

    ``title`` goes to ``st.header`` unchanged, so pages keep their exact title text.
    ``description`` is Markdown body text; ``chips`` are ``(label, kind)`` pairs.
    """
    if eyebrow:
        st.html(eyebrow_html(eyebrow))
    st.header(title, anchor=anchor)
    if chips:
        status_chips(chips)
    if description:
        st.markdown(description)


def breadcrumbs_html(parts: Sequence[str]) -> str:
    """Accessible, non-interactive context trail; the last part is the current page."""
    items = []
    for index, part in enumerate(parts):
        current = ' aria-current="page"' if index == len(parts) - 1 else ""
        separator = (
            f'<span class="{CSS_PREFIX}-breadcrumbs-sep" aria-hidden="true">/</span>'
            if index
            else ""
        )
        items.append(f"<li{current}>{separator}{html_text(part)}</li>")
    return f'<nav class="{CSS_PREFIX}-breadcrumbs" aria-label="Breadcrumb"><ol>{"".join(items)}</ol></nav>'


def breadcrumbs(parts: Sequence[str]) -> None:
    """Render ``CSc 8830 / Module 5-6 / Motion Tracking`` style context.

    Parts are plain text, not links: the dashboards have no URL routing for pages, so a
    link would promise navigation that does not exist.
    """
    parts = [str(p) for p in parts if str(p).strip()]
    if parts:
        st.html(breadcrumbs_html(parts))


def section_header(
    title: str, *, description: str | None = None, anchor: str | None = None
) -> None:
    """Native ``st.subheader`` with optional short supporting copy as a caption."""
    st.subheader(title, anchor=anchor)
    if description:
        st.caption(description)


def footer(parts: Sequence[str] = DEFAULT_FOOTER) -> None:
    """Minimal closing line, for example course name and institution."""
    st.divider()
    st.caption(" · ".join(str(p) for p in parts))
