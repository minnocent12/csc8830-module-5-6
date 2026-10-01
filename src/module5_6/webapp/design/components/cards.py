"""Bordered card surfaces built on ``st.container(border=True)``.

Cards are presentation only. Every widget, key, default, run button, and execution gate
stays in the page that calls them:

    with configuration_card(caption="Parameters for the Lucas-Kanade tracker"):
        with parameter_group("Window", help="Search window around each feature."):
            win = st.number_input("Window size", value=21, key="lk_win")
        run = st.button("Run tracking", type="primary")
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from ._text import markdown_text


@contextmanager
def card(
    title: str | None = None, *, caption: str | None = None, key: str | None = None
) -> Iterator[DeltaGenerator]:
    """A bordered container with an optional h4 title and caption; yields the container."""
    container = st.container(border=True, key=key)
    with container:
        if title:
            st.markdown(f"#### {markdown_text(title)}")
        if caption:
            st.caption(caption)
        yield container


@contextmanager
def configuration_card(
    title: str = "Configuration", *, caption: str | None = None, key: str | None = None
) -> Iterator[DeltaGenerator]:
    """The card that holds an experiment's controls. The page still owns every control."""
    with card(title, caption=caption, key=key) as container:
        yield container


@contextmanager
def parameter_group(title: str, *, help: str | None = None) -> Iterator[DeltaGenerator]:
    """Label and optional explanation for a set of related native inputs inside a card."""
    container = st.container()
    with container:
        st.markdown(f"**{markdown_text(title)}**")
        if help:
            st.caption(help)
        yield container
