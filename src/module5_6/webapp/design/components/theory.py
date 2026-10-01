"""Theory and derivation structure: numbered sections and equation blocks.

Equations go to native ``st.latex`` exactly as written; nothing here parses, rewrites, or
reformats LaTeX. Theory prose is not forced into cards: a section is a heading plus an
ordinary container the page fills with Markdown, equations, and figures.
"""
from __future__ import annotations

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from ._text import markdown_text

SECTION_SEPARATOR = " · "  # middle dot, as in "01 · Projection model"


def section_label(title: str, number: int | str | None = None) -> str:
    """Heading text: integers are zero padded ("01"), strings ("2.1") are kept as given."""
    if number is None:
        return title
    prefix = f"{number:02d}" if isinstance(number, int) else str(number)
    return f"{prefix}{SECTION_SEPARATOR}{title}"


def theory_section(
    title: str,
    *,
    number: int | str | None = None,
    body: str | None = None,
    anchor: str | None = None,
) -> DeltaGenerator:
    """Render a numbered ``st.subheader`` and optional Markdown body; return a container.

    The return value can be used as a context manager to keep the section's content
    together, or ignored and the page simply continues writing below the heading.
    """
    st.subheader(section_label(title, number), anchor=anchor)
    container = st.container()
    if body:
        container.markdown(body)
    return container


def equation_block(latex: str, *, title: str | None = None, caption: str | None = None) -> None:
    """A bordered block holding one native ``st.latex`` equation, unchanged."""
    with st.container(border=True):
        if title:
            st.markdown(f"**{markdown_text(title)}**")
        st.latex(latex)
        if caption:
            st.caption(caption)
