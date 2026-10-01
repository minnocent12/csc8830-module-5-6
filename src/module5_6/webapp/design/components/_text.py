"""Escaping helpers for caller-supplied text. No Streamlit here.

Two different targets need two different escapes:
* ``html_text`` for the few small elements rendered with ``st.html`` (eyebrow, breadcrumbs,
  chips). Everything is entity-escaped, so caller text can never become markup or script.
* ``markdown_text`` for titles placed inside a Markdown heading or bold span, so characters
  such as ``_`` in ``view_2`` or ``$`` do not turn into italics or LaTeX.
Body text passed to ``st.markdown`` on purpose (descriptions, explanations) is left as
Markdown, because pages already write Markdown there.
"""
from __future__ import annotations

import html
import re

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+!|~$])")


def html_text(text: object) -> str:
    """Entity-escape any value for use as HTML text or attribute content."""
    return html.escape(str(text), quote=True)


def markdown_text(text: object) -> str:
    """Backslash-escape Markdown control characters so text renders literally."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", str(text))
