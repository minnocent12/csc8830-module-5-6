"""The committed Streamlit config and dependency floor must match the vendored design kit.

``src/module5_6/webapp/design/`` is a vendored copy of the csc8830-ui kit, and
``.streamlit/config.toml`` is generated from it. Neither is edited by hand.
"""
from __future__ import annotations

import re
from pathlib import Path

from module5_6.webapp.design import KIT_VERSION, STREAMLIT_REQUIREMENT, render_config_toml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_config_matches_vendored_design_kit() -> None:
    config = REPO_ROOT / ".streamlit" / "config.toml"
    assert config.read_text(encoding="utf-8") == render_config_toml()


def test_vendored_design_kit_has_a_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", KIT_VERSION)


def test_streamlit_dependency_floor_matches_design_kit() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"{STREAMLIT_REQUIREMENT}"' in pyproject


def test_vendored_components_import() -> None:
    from module5_6.webapp.design import components

    for name in ("page_header", "metric_row", "status_banner", "image_comparison", "upload_panel"):
        assert callable(getattr(components, name))


def test_page_slugs_are_unique() -> None:
    from module5_6.webapp.design.navigation import build_navigation
    from module5_6.webapp.pages import get_pages
    from module5_6.webapp.registry import collect_pages

    (module,) = build_navigation(collect_pages([get_pages]))
    assert len({page.slug for page in module.pages}) == len(module.pages)


def test_standalone_shell_shows_module_context_without_a_selectbox() -> None:
    from streamlit.testing.v1 import AppTest

    from module5_6.webapp.pages import get_pages

    app = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert not app.sidebar.selectbox  # pages may have their own selectboxes
    assert any(m.value == "**Module 5-6**" for m in app.sidebar.markdown)
    assert app.sidebar.radio[0].label == "Page"
    assert app.sidebar.radio[0].options == [p.page_label for p in get_pages()]
    assert any("Breadcrumb" in e.proto.body for e in app.get("html"))
