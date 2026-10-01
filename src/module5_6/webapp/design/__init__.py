"""CSc 8830 design kit: canonical design tokens, Streamlit theme, and global styles.

This package is the canonical source that assignment repositories vendor into their own
``src/<module>/webapp/design/`` package. Assignment code never imports it at runtime.
Importing the package never imports Streamlit; only ``inject_global_styles()`` does.
"""
from __future__ import annotations

from .accessibility import ContrastUse, contrast_ratio, relative_luminance
from .styles import CSS_PREFIX, build_css, css_variables, inject_global_styles
from .theme import STREAMLIT_REQUIREMENT, render_config_toml, streamlit_theme
from .tokens import (
    BORDERS,
    BREAKPOINTS,
    COLORS,
    CONTRAST_PAIRS,
    ELEVATION,
    FONT_FAMILY_MONO,
    FONT_FAMILY_SANS,
    HEADINGS,
    LAYOUT,
    RADII,
    SPACING,
    TYPOGRAPHY,
    color_roles,
)
from .version import KIT_VERSION

__all__ = [
    "BORDERS",
    "BREAKPOINTS",
    "COLORS",
    "CONTRAST_PAIRS",
    "CSS_PREFIX",
    "ELEVATION",
    "FONT_FAMILY_MONO",
    "FONT_FAMILY_SANS",
    "HEADINGS",
    "KIT_VERSION",
    "LAYOUT",
    "RADII",
    "SPACING",
    "STREAMLIT_REQUIREMENT",
    "TYPOGRAPHY",
    "ContrastUse",
    "build_css",
    "color_roles",
    "contrast_ratio",
    "css_variables",
    "inject_global_styles",
    "relative_luminance",
    "render_config_toml",
    "streamlit_theme",
]
