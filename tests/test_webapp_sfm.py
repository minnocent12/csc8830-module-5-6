from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

# Absolute, so the path means the same thing on every supported Streamlit version: 1.47
# resolves a relative path against the working directory first, 1.64 against this file.
APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def test_sfm_page_renders_bundled_real_results_without_exception() -> None:
    """With no images uploaded, the SfM page must show the completed real experiment, not the
    pending-experiment banner - see root AGENTS.md "Bundled real-sample fallback"."""
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Structure From Motion").run()

    assert not app.exception
    markdown_text = " ".join(item.value for item in app.markdown)
    assert "PENDING USER EXPERIMENT" not in markdown_text
    assert "reference view" in markdown_text.lower() or any(
        "Reference view" in item.value for item in app.markdown
    )
    subheaders = [item.value for item in app.subheader]
    assert any("Boundary reconstruction" in value for value in subheaders)


def test_experiments_page_reports_completed_sfm_experiment() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Experiments & Results").run()

    assert not app.exception
    success_text = " ".join(item.value for item in app.success)
    assert "Four-view SfM experiment complete" in success_text
