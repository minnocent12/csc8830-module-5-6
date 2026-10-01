"""Reusable Streamlit presentation components for the CSc 8830 application.

Import from a vendored copy, for example ``from module5_6.webapp.design.components import
page_header, metric_row``. This subpackage imports Streamlit; the parent package does not,
so tokens and theme stay usable without it.

Principles shared by every component:
* wrap native Streamlit elements (headers, metrics, alerts, images, dataframes, uploaders)
  so behavior, accessibility, and AppTest visibility are unchanged
* presentation only: no image processing, no color conversion, no metric computation, and
  no judgement of results
* never create, rename, or default widget keys; the page passes its own
* caller text is shown as given; text placed in custom HTML is entity-escaped
"""
from __future__ import annotations

from ._grid import balanced_row_sizes
from .cards import card, configuration_card, parameter_group
from .data import data_table, download_action
from .images import ImageItem, image_card, image_comparison, image_gallery
from .layout import breadcrumbs, eyebrow, footer, page_header, section_header
from .metrics import MAX_METRIC_COLUMNS, MetricSpec, metric_card, metric_row
from .results import RESULT_SECTION_ORDER, experiment_summary, result_section
from .status import (
    StatusKind,
    empty_state,
    pending_state,
    status_banner,
    status_chip,
    status_chips,
    success_state,
)
from .theory import equation_block, theory_section
from .uploads import upload_panel

__all__ = [
    "MAX_METRIC_COLUMNS",
    "RESULT_SECTION_ORDER",
    "ImageItem",
    "MetricSpec",
    "StatusKind",
    "balanced_row_sizes",
    "breadcrumbs",
    "card",
    "configuration_card",
    "data_table",
    "download_action",
    "empty_state",
    "equation_block",
    "experiment_summary",
    "eyebrow",
    "footer",
    "image_card",
    "image_comparison",
    "image_gallery",
    "metric_card",
    "metric_row",
    "page_header",
    "parameter_group",
    "pending_state",
    "result_section",
    "section_header",
    "status_banner",
    "status_chip",
    "status_chips",
    "success_state",
    "theory_section",
    "upload_panel",
]
