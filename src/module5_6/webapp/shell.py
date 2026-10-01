"""Streamlit shell for the Module 5-6 app: a thin adapter over the shared csc8830-ui shell.

Navigation, identity, breadcrumbs, and footer come from ``module5_6.webapp.design.shell``,
a vendored copy of the course design kit. This module only states what is specific here.
"""
from __future__ import annotations

from collections.abc import Sequence

from module5_6.webapp._page import PageSpec
from module5_6.webapp.design.shell import render_shell


def render_app(pages: Sequence[PageSpec], *, title: str = "CSc 8830 - Module 5-6") -> None:
    """Render the standalone Module 5-6 app around the selected page."""
    render_shell(pages, page_title=title, standalone=True)
