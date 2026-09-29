"""Dashboard-compatible page contract for Module 5-6."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

RenderFn = Callable[[], None]


@dataclass(frozen=True)
class PageSpec:
    """One selectable page contributed by Module 5-6."""

    module_label: str
    page_label: str
    order: int
    render: RenderFn
