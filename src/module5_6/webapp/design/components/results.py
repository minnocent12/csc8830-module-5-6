"""Result presentation: experiment summary card and consistent result sections.

Nothing here judges a result. ``experiment_summary`` requires the caller to state the
status explicitly; it never infers success or failure from metric values.

Suggested section order for experiment pages, a convention rather than a rule:
Configuration, Evidence, Metrics, Validation, Interpretation (``RESULT_SECTION_ORDER``).
"""
from __future__ import annotations

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from .cards import card
from .layout import section_header
from .status import StatusKind, status_chip

RESULT_SECTION_ORDER = ("Configuration", "Evidence", "Metrics", "Validation", "Interpretation")


def experiment_summary(
    *,
    method: str,
    evidence: str,
    status_label: str,
    status_kind: StatusKind | str,
    result: str | None = None,
    interpretation: str | None = None,
    title: str = "Experiment summary",
) -> None:
    """Card with an explicit status chip and short Markdown fields.

    ``status_label`` and ``status_kind`` are required: the page decides what the status is.
    """
    with card(title):
        status_chip(status_label, status_kind)
        fields = [("Method", method), ("Evidence", evidence), ("Result", result),
                  ("Interpretation", interpretation)]
        for name, text in fields:
            if text:
                st.markdown(f"**{name}.** {text}")


def result_section(
    title: str, *, description: str | None = None, anchor: str | None = None
) -> DeltaGenerator:
    """A native section heading with optional caption; returns a container for its content."""
    section_header(title, description=description, anchor=anchor)
    return st.container()
