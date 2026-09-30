from __future__ import annotations

from pathlib import Path

import module5_6.webapp.pages as pages


def test_optical_flow_bundled_results_render_with_real_committed_data() -> None:
    assert pages._render_optical_flow_bundled_results() is True


def test_optical_flow_bundled_results_absent_returns_false(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pages, "_OPTICAL_FLOW_RESULTS_DIR", tmp_path / "does_not_exist")
    assert pages._render_optical_flow_bundled_results() is False


def test_tracking_bundled_results_render_with_real_committed_data() -> None:
    assert pages._render_tracking_bundled_results() is True


def test_tracking_bundled_results_absent_returns_false(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pages, "_TRACKING_RESULTS_DIR", tmp_path / "does_not_exist")
    assert pages._render_tracking_bundled_results() is False
