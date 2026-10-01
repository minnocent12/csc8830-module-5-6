"""Metric cards and rows around native ``st.metric``.

Labels and values are passed through exactly as given: no case change, no rounding, no
reformatting, so units such as ``PSNR (dB)`` or ``(px)`` keep their meaning and AppTest still
reads every value from ``app.metric``. Long labels wrap (global styles) rather than clip.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import streamlit as st

from ._grid import grid_rows

MAX_METRIC_COLUMNS = 4


@dataclass(frozen=True)
class MetricSpec:
    """One metric. ``value`` and ``delta`` are shown exactly as supplied."""

    label: str
    value: Any
    delta: Any = None
    help: str | None = None
    delta_color: str = "normal"


def metric_card(
    label: str,
    value: Any,
    delta: Any = None,
    *,
    help: str | None = None,
    delta_color: str = "normal",
) -> None:
    """A single native ``st.metric``; same arguments, same semantics."""
    st.metric(label, value, delta=delta, delta_color=delta_color, help=help)


def _spec(item: MetricSpec | Sequence[Any]) -> MetricSpec:
    return item if isinstance(item, MetricSpec) else MetricSpec(*item)


def metric_row(
    metrics: Sequence[MetricSpec | Sequence[Any]], *, max_columns: int = MAX_METRIC_COLUMNS
) -> None:
    """Lay out metrics in balanced rows of at most four columns.

    One to four metrics share a single row; five or more wrap onto further rows of nearly
    equal length (5 becomes 3 and 2, 7 becomes 4 and 3), so nothing is squeezed into five
    narrow columns. Items may be ``MetricSpec`` or ``(label, value[, delta])`` tuples.
    """
    if not 1 <= max_columns <= MAX_METRIC_COLUMNS:
        raise ValueError(f"max_columns must be between 1 and {MAX_METRIC_COLUMNS}")
    specs = [_spec(item) for item in metrics]
    columns, rows = grid_rows(specs, max_columns)
    for row in rows:
        for column, spec in zip(st.columns(columns), row):
            with column:
                metric_card(
                    spec.label,
                    spec.value,
                    spec.delta,
                    help=spec.help,
                    delta_color=spec.delta_color,
                )
