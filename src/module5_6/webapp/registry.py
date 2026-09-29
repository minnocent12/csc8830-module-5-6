"""Provider-agnostic page collection for standalone and shared hosts."""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

from module5_6.webapp._page import PageSpec

PageProvider = Callable[[], Iterable[PageSpec]]


def collect_pages(providers: Sequence[PageProvider]) -> list[PageSpec]:
    """Merge, validate, de-duplicate, and stably order page providers."""
    merged: dict[tuple[str, str], PageSpec] = {}
    for provider in providers:
        for page in provider():
            if not isinstance(page, PageSpec):
                raise TypeError(f"provider {provider!r} yielded {type(page)!r}, expected PageSpec")
            merged[(page.module_label, page.page_label)] = page
    return sorted(merged.values(), key=lambda page: (page.module_label, page.order, page.page_label))
