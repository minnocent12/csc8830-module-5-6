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
    """Small uppercase context label above a page title, for example a module name.

    The label is shown uppercase, so never pass text whose case matters (such as "CSc 8830").
    """
    return f'<p class="{CSS_PREFIX}-eyebrow">{html_text(text)}</p>'


def eyebrow(text: str) -> None:
    """Render an eyebrow label on its own, for example above a card title."""
    st.html(eyebrow_html(text))


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


Crumb = str | tuple[str, str]  # plain text, or (text, same-app query link such as "?module=home")


def _crumb(part: Crumb) -> tuple[str, str | None]:
    if isinstance(part, tuple):
        label, href = part
        if not str(href).startswith("?"):
            raise ValueError(f"breadcrumb links must be same-app query strings, got {href!r}")
        return str(label), str(href)
    return str(part), None


def breadcrumbs_html(parts: Sequence[Crumb]) -> str:
    """Accessible context trail; the last part is the current page and is never a link."""
    items = []
    for index, part in enumerate(parts):
        label, href = _crumb(part)
        last = index == len(parts) - 1
        current = ' aria-current="page"' if last else ""
        separator = (
            f'<span class="{CSS_PREFIX}-breadcrumbs-sep" aria-hidden="true">/</span>'
            if index
            else ""
        )
        text = html_text(label)
        if href and not last:
            text = f'<a href="{html_text(href)}" target="_self">{text}</a>'
        items.append(f"<li{current}>{separator}{text}</li>")
    return f'<nav class="{CSS_PREFIX}-breadcrumbs" aria-label="Breadcrumb"><ol>{"".join(items)}</ol></nav>'


def breadcrumbs(parts: Sequence[Crumb]) -> None:
    """Render ``CSc 8830 / Module 5-6 / Motion Tracking`` style context.

    A part is plain text, or ``(text, href)`` when a genuine destination exists; ``href``
    must be a same-app query string (it starts with ``?``). The current (last) part is never
    a link. Following a link reloads the app at that URL, which starts a new session.
    """
    parts = [p for p in parts if _crumb(p)[0].strip()]
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
