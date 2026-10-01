"""Row layout arithmetic shared by metric rows and image galleries. No Streamlit here."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def balanced_row_sizes(count: int, max_columns: int) -> list[int]:
    """Split ``count`` items into the fewest rows of at most ``max_columns``, as evenly as possible.

    Rows differ in length by at most one, longer rows first: 5 items with a maximum of 4
    become ``[3, 2]`` rather than ``[4, 1]``, so no row is a single stretched item.
    """
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1")
    if count <= 0:
        return []
    rows = math.ceil(count / max_columns)
    base, extra = divmod(count, rows)
    return [base + 1] * extra + [base] * (rows - extra)


def grid_rows(items: Sequence[T], max_columns: int) -> tuple[int, list[list[T]]]:
    """Group ``items`` into balanced rows and return ``(columns_per_row, rows)``.

    Every row is laid out with the same column count (the longest row), so cells line up
    vertically and a shorter final row leaves empty space instead of widening its items.
    """
    sizes = balanced_row_sizes(len(items), max_columns)
    rows: list[list[T]] = []
    start = 0
    for size in sizes:
        rows.append(list(items[start : start + size]))
        start += size
    return (sizes[0] if sizes else 0), rows
