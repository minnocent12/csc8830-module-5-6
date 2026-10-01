"""WCAG 2.x contrast helpers used to verify design token pairings.

Relative luminance and contrast ratio follow WCAG 2.2, Success Criteria 1.4.3 and 1.4.11:

    channel c in [0, 1]:  c_lin = c / 12.92                    if c <= 0.04045
                          c_lin = ((c + 0.055) / 1.055) ** 2.4  otherwise
    L = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin
    ratio = (L_lighter + 0.05) / (L_darker + 0.05),   range 1.0 to 21.0
"""
from __future__ import annotations

import re
from enum import Enum

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


class ContrastUse(str, Enum):
    """What a foreground/background pairing is used for, and therefore its WCAG minimum."""

    NORMAL_TEXT = "normal_text"  # SC 1.4.3 AA: body, captions, button labels
    LARGE_TEXT = "large_text"  # SC 1.4.3 AA: at least 24px regular or 18.66px bold
    NON_TEXT = "non_text"  # SC 1.4.11 AA: focus rings, input outlines, state indicators
    DECORATIVE = "decorative"  # no requirement: carries no information on its own


MINIMUM_RATIO: dict[ContrastUse, float] = {
    ContrastUse.NORMAL_TEXT: 4.5,
    ContrastUse.LARGE_TEXT: 3.0,
    ContrastUse.NON_TEXT: 3.0,
    ContrastUse.DECORATIVE: 1.0,
}


def is_hex_color(value: str) -> bool:
    """Return True for a ``#RRGGBB`` color string."""
    return bool(_HEX_COLOR.match(value))


def relative_luminance(hex_color: str) -> float:
    """Relative luminance of an sRGB ``#RRGGBB`` color, in [0, 1]."""
    if not is_hex_color(hex_color):
        raise ValueError(f"expected #RRGGBB, got {hex_color!r}")
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two ``#RRGGBB`` colors (order does not matter)."""
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)
