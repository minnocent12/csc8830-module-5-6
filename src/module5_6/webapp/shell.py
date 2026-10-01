"""Streamlit rendering shell for the Module 5-6 page provider."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import streamlit as st

from module5_6.webapp._page import PageSpec
from module5_6.webapp.design import inject_global_styles


def render_app(pages: Sequence[PageSpec], *, title: str = "CSc 8830 - Module 5-6") -> None:
    """Render a sidebar page selector and dispatch the selected page."""
    st.set_page_config(page_title=title, layout="wide")
    inject_global_styles()
    if not pages:
        st.error("No pages registered.")
        return

    by_module: dict[str, list[PageSpec]] = defaultdict(list)
    for page in pages:
        by_module[page.module_label].append(page)

    with st.sidebar:
        st.title(title)
        module_label = st.selectbox("Module", list(by_module))
        module_pages = by_module[module_label]
        page_label = st.radio("Page", [page.page_label for page in module_pages])

    selected = next(page for page in module_pages if page.page_label == page_label)
    selected.render()
