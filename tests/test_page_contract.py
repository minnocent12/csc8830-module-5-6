from __future__ import annotations

import pytest

from module5_6.webapp._page import PageSpec
from module5_6.webapp.pages import get_pages
from module5_6.webapp.registry import collect_pages


def _fake_provider() -> list[PageSpec]:
    return [
        PageSpec("Module X", "Overview", 10, lambda: None),
        PageSpec("Module X", "Details", 20, lambda: None),
    ]


def test_module5_6_provider_returns_stable_pages() -> None:
    pages = collect_pages([get_pages])
    assert [page.page_label for page in pages] == [
        "Optical Flow",
        "Motion Tracking",
        "Bilinear Interpolation & Theory",
        "Structure From Motion",
        "Experiments & Results",
    ]
    assert {page.module_label for page in pages} == {"Module 5-6"}


def test_registry_merges_and_orders_multiple_providers() -> None:
    pages = collect_pages([get_pages, _fake_provider])
    assert [(page.module_label, page.page_label) for page in pages] == [
        ("Module 5-6", "Optical Flow"),
        ("Module 5-6", "Motion Tracking"),
        ("Module 5-6", "Bilinear Interpolation & Theory"),
        ("Module 5-6", "Structure From Motion"),
        ("Module 5-6", "Experiments & Results"),
        ("Module X", "Overview"),
        ("Module X", "Details"),
    ]


def test_later_provider_overrides_duplicate_page() -> None:
    marker = object()

    def override() -> list[PageSpec]:
        return [PageSpec("Module 5-6", "Experiments & Results", 50, lambda: marker)]

    experiments = next(
        page
        for page in collect_pages([get_pages, override])
        if page.page_label == "Experiments & Results"
    )
    assert experiments.render() is marker


def test_registry_rejects_non_pagespec() -> None:
    with pytest.raises(TypeError):
        collect_pages([lambda: ["not a page"]])
