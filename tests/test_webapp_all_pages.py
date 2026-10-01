from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Absolute, so the path means the same thing on every supported Streamlit version: 1.47
# resolves a relative path against the working directory first, 1.64 against this file.
APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

_PAGE_LABELS = [
    "Optical Flow",
    "Motion Tracking",
    "Bilinear Interpolation & Theory",
    "Structure From Motion",
    "Experiments & Results",
]


@pytest.mark.parametrize("page_label", _PAGE_LABELS)
def test_page_renders_without_exception(page_label: str) -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.sidebar.radio[0].set_value(page_label).run()
    assert not app.exception


def test_optical_flow_page_shows_real_bundled_statistics_with_no_upload() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Optical Flow").run()
    assert not app.exception
    mean_flow_values = [m.value for m in app.metric if m.label == "Mean |flow| (px)"]
    assert "0.277" in mean_flow_values  # video_1, docs/EXPERIMENTAL_RESULTS.md Section 2
    assert "0.383" in mean_flow_values  # video_2, docs/EXPERIMENTAL_RESULTS.md Section 2


def test_motion_tracking_page_shows_real_pixel_errors_with_no_upload() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Motion Tracking").run()
    assert not app.exception
    pixel_errors = [m.value for m in app.metric if m.label == "Pixel error e (px)"]
    assert "4.628" in pixel_errors
    assert "8.408" in pixel_errors


def test_experiments_page_reports_all_experiments_complete() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Experiments & Results").run()
    assert not app.exception
    success_text = " ".join(s.value for s in app.success)
    assert "Video 1: experiment complete." in success_text
    assert "Video 2: experiment complete." in success_text
    assert "Four-view SfM experiment complete" in success_text
    warning_text = " ".join(w.value for w in app.warning)
    assert "PENDING" not in warning_text
