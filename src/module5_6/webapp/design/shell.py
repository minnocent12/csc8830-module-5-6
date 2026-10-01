"""Shared application shell: identity, native navigation, breadcrumbs, footer, URL sync.

Every app (the combined dashboards and each standalone module) calls ``render_shell``
from a thin adapter. The shell knows nothing about computer vision or any assignment; it
takes pages with the structural contract ``module_label``, ``page_label``, ``order``,
``render`` and calls the selected page's ``render()`` unchanged.

Navigation stays native Streamlit: a module ``st.selectbox`` (combined dashboards only) and
a page ``st.radio``, which is always the first radio in the app so existing AppTest flows
keep working. A standalone app shows its single module as plain text instead of a
one-option selectbox.

URL synchronization (``sync_query_params=True``, combined dashboards only) owns exactly two
query parameters, ``module`` and ``page``, holding stable slugs (see ``navigation``):

1. On the first run of a browser session the URL is read once and resolved, with fallbacks,
   into the widgets' session state, before the widgets are created.
2. From then on the widgets are the single source of truth.
3. After the widgets render, ``module`` and ``page`` are written back only when they differ
   from the current selection. Other query parameters are left untouched.

Because the URL is read once and written only on change, the two never fight and no rerun
loop is possible. A refresh or a copied link starts a new session, so step 1 restores the
page. The Home view of a later phase can reuse this flow under the reserved slug ``home``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import streamlit as st

from .components._text import html_text, markdown_text
from .components.layout import breadcrumbs, footer
from .navigation import NavModule, NavPage, build_navigation, module_by_label, resolve
from .styles import CSS_PREFIX, inject_global_styles

QUERY_MODULE = "module"
QUERY_PAGE = "page"
MODULE_STATE_KEY = f"{CSS_PREFIX}_nav_module"
URL_READ_STATE_KEY = f"{CSS_PREFIX}_nav_url_read"
DEFAULT_FOOTER = ("CSc 8830 Computer Vision", "Georgia State University")

PageContext = Callable[[Any], str | None]


def page_state_key(module: NavModule) -> str:
    """Session-state key of a module's page radio; one per module, so each remembers its page."""
    return f"{CSS_PREFIX}_nav_page__{module.slug}"


def identity_html(title: str, subtitle: str) -> str:
    """Sidebar identity block. Text is escaped; case is preserved exactly ("CSc")."""
    return (
        f'<div class="{CSS_PREFIX}-identity">'
        f'<h1 class="{CSS_PREFIX}-identity-title">{html_text(title)}</h1>'
        f'<p class="{CSS_PREFIX}-identity-subtitle">{html_text(subtitle)}</p>'
        "</div>"
    )


def nav_label_html(text: str) -> str:
    """Quiet uppercase category label such as MODULE or PAGES."""
    return f'<p class="{CSS_PREFIX}-nav-label">{html_text(text)}</p>'


def _read_url_once(modules: Sequence[NavModule]) -> None:
    """Seed widget state from the URL on the session's first run only."""
    if st.session_state.get(URL_READ_STATE_KEY):
        return
    st.session_state[URL_READ_STATE_KEY] = True
    module, page = resolve(
        modules, st.query_params.get(QUERY_MODULE), st.query_params.get(QUERY_PAGE)
    )
    st.session_state[MODULE_STATE_KEY] = module.label
    st.session_state[page_state_key(module)] = page.label


def _write_url(module: NavModule, page: NavPage) -> None:
    """Write only the two owned keys, and only when they changed."""
    if st.query_params.get(QUERY_MODULE) != module.slug:
        st.query_params[QUERY_MODULE] = module.slug
    if st.query_params.get(QUERY_PAGE) != page.slug:
        st.query_params[QUERY_PAGE] = page.slug


def _drop_stale_state(key: str, valid: Sequence[str]) -> None:
    if key in st.session_state and st.session_state[key] not in valid:
        del st.session_state[key]


def _select_module(modules: Sequence[NavModule], standalone: bool) -> NavModule:
    st.html(nav_label_html("Module"))
    if standalone:
        st.markdown(f"**{markdown_text(modules[0].label)}**")
        return modules[0]
    labels = [m.label for m in modules]
    _drop_stale_state(MODULE_STATE_KEY, labels)
    label = st.selectbox("Module", labels, key=MODULE_STATE_KEY, label_visibility="collapsed")
    return module_by_label(modules, label) or modules[0]


def _select_page(module: NavModule) -> NavPage:
    st.html(nav_label_html("Pages"))
    labels = [p.label for p in module.pages]
    key = page_state_key(module)
    remembered = f"{key}__last"
    _drop_stale_state(key, labels)
    # Streamlit discards a widget's state on runs where it is not drawn, which is every run
    # while another module is selected. A plain session value keeps each module's last page.
    if key not in st.session_state and st.session_state.get(remembered) in labels:
        st.session_state[key] = st.session_state[remembered]
    label = st.radio("Page", labels, key=key, label_visibility="collapsed")
    st.session_state[remembered] = label
    return module.page_by_label(label) or module.pages[0]


def render_shell(
    pages: Sequence[Any],
    *,
    page_title: str,
    app_title: str = "CSc 8830",
    app_subtitle: str = "Computer Vision",
    standalone: bool = False,
    page_context: PageContext | None = None,
    sync_query_params: bool = False,
    notices: Sequence[str] = (),
    empty_message: str = "No pages registered.",
    footer_parts: Sequence[str] = DEFAULT_FOOTER,
) -> None:
    """Render the whole app around the selected page.

    Args:
        pages: page objects, already ordered (module, order, label).
        page_title: browser tab title passed to ``st.set_page_config``.
        app_title, app_subtitle: sidebar identity, shown exactly as given.
        standalone: True for a single-module app; the module is shown as text, not a
            selectbox. Raises ``ValueError`` if the pages span several modules.
        page_context: optional ``callable(page) -> str | None``; a returned string is shown
            as a sidebar caption under the page list.
        sync_query_params: keep ``?module=...&page=...`` in step with the selection
            (combined dashboards only; ignored when ``standalone``).
        notices: dashboard notices shown in a collapsed sidebar expander.
        empty_message: error shown when no pages are registered.
        footer_parts: footer text segments.
    """
    st.set_page_config(page_title=page_title, layout="wide")
    inject_global_styles()
    if not pages:
        st.error(empty_message)
        for notice in notices:
            st.info(notice)
        return

    modules = build_navigation(pages)
    if standalone and len(modules) != 1:
        raise ValueError(f"a standalone app needs exactly one module, got {len(modules)}")
    sync = sync_query_params and not standalone
    if sync:
        _read_url_once(modules)

    with st.sidebar:
        st.html(identity_html(app_title, app_subtitle))
        module = _select_module(modules, standalone)
        page = _select_page(module)
        context = page_context(page.page) if page_context else None
        if context:
            st.divider()
            st.caption(context)
        if notices:
            with st.expander("Dashboard notices"):
                for notice in notices:
                    st.caption(notice)

    if sync:
        _write_url(module, page)

    breadcrumbs([app_title, module.label, page.label])
    page.page.render()
    footer(footer_parts)
