"""Minimal global CSS for gaps the native Streamlit theme cannot express.

Policy: the native theme (``theme.py``) does almost everything. CSS here is limited to the
rules below, each documented with its selector, reason, and fragility. Custom properties
and any future classes use the ``csc8830-`` namespace.

Streamlit internals are reached only through ``data-testid`` attributes, never through the
generated ``st-emotion-cache-*`` class names, which change between releases. ``data-testid``
values are Streamlit test hooks, not a public API; each one used here was confirmed present
in Streamlit 1.47, 1.49, and 1.64.

| Rule | Selector | Reason | Fragility |
|---|---|---|---|
| 1 | ``:root`` | expose tokens as ``--csc8830-*`` custom properties for later components | none |
| 2 | ``[data-testid="stMainBlockContainer"]`` | cap content width; wide layout otherwise runs edge to edge on large monitors | low: test hook renamed once (1.4x) |
| 3 | ``.stApp :is(h1, ..., h6)`` | heading color; the theme has one text color for everything | low: plain tags under the app root class |
| 4 | ``[data-testid="stCaptionContainer"]`` | Streamlit fades captions with ``opacity: 0.6``, which drops them below AA; restore full opacity and use the muted text color (6.21:1) | low |
| 5 | ``[data-testid="stMetricLabel"]`` and its ``div``/``p`` descendants, ``[data-testid="stMetricValue"]`` | metric label and value typography; native theme keys for this arrived after 1.47. The inner paragraph sets its own size, and the inner elements truncate with an ellipsis (a different element in 1.47 and 1.64), which hides long labels such as "Mean reproj. error, inliers (px)" on laptop widths, so labels wrap instead. Labels are never case-transformed because they carry units | medium: depends on the label's inner markup, which already differs between 1.47 and 1.64 |
| 6 | ``.stApp :is(a, button, summary, [role="tab"]):focus-visible`` | one visible keyboard focus ring in the accessible orange | none: standard pseudo-class |

Nothing here hides a control, changes layout order, or replaces a native element, so
AppTest and assistive technology see the same element tree as before.
"""
from __future__ import annotations

from dataclasses import fields

from .tokens import (
    BORDERS,
    COLORS,
    FONT_FAMILY_MONO,
    FONT_FAMILY_SANS,
    LAYOUT,
    RADII,
    SPACING,
    TYPOGRAPHY,
    ColorTokens,
    RadiusTokens,
    SpacingTokens,
)
from .version import KIT_VERSION

CSS_PREFIX = "csc8830"
STYLE_ELEMENT_ID = f"{CSS_PREFIX}-global-styles"


def _kebab(name: str) -> str:
    return name.replace("_", "-")


def css_variables() -> dict[str, str]:
    """Design tokens as ``{"--csc8830-...": value}`` custom properties."""
    variables: dict[str, str] = {}
    for f in fields(ColorTokens):
        variables[f"--{CSS_PREFIX}-color-{_kebab(f.name)}"] = getattr(COLORS, f.name)
    for f in fields(SpacingTokens):
        variables[f"--{CSS_PREFIX}-space-{f.name}"] = f"{getattr(SPACING, f.name)}px"
    for f in fields(RadiusTokens):
        variables[f"--{CSS_PREFIX}-radius-{_kebab(f.name)}"] = f"{getattr(RADII, f.name)}px"
    variables[f"--{CSS_PREFIX}-font-sans"] = FONT_FAMILY_SANS
    variables[f"--{CSS_PREFIX}-font-mono"] = FONT_FAMILY_MONO
    variables[f"--{CSS_PREFIX}-content-max-width"] = f"{LAYOUT.content_max_width}px"
    variables[f"--{CSS_PREFIX}-reading-max-width"] = f"{LAYOUT.reading_max_width}px"
    return variables


def build_css() -> str:
    """Return the global stylesheet text. Deterministic for a given KIT_VERSION."""
    p = CSS_PREFIX
    label = TYPOGRAPHY.metric_label
    value = TYPOGRAPHY.metric_value
    root = "\n".join(f"  {name}: {val};" for name, val in css_variables().items())
    return f"""/* csc8830-ui {KIT_VERSION} global styles */
:root {{
{root}
}}
[data-testid="stMainBlockContainer"] {{
  max-width: var(--{p}-content-max-width);
  margin-left: auto;
  margin-right: auto;
}}
.stApp :is(h1, h2, h3, h4, h5, h6) {{
  color: var(--{p}-color-text-heading);
}}
[data-testid="stCaptionContainer"] {{
  color: var(--{p}-color-text-muted);
  opacity: 1;
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {{
  color: var(--{p}-color-text-muted);
  font-size: {label.size_px}px;
  font-weight: {label.weight};
  letter-spacing: {label.letter_spacing_em}em;
  text-transform: {"uppercase" if label.uppercase else "none"};
}}
[data-testid="stMetricLabel"] :is(div, p) {{
  white-space: normal;
  overflow: visible;
  text-overflow: clip;
}}
[data-testid="stMetricValue"] {{
  color: var(--{p}-color-text-heading);
  font-size: {value.size_px}px;
  font-weight: {value.weight};
  line-height: {value.line_height};
  font-variant-numeric: tabular-nums;
}}
.stApp :is(a, button, summary, [role="tab"]):focus-visible {{
  outline: {BORDERS.focus_ring_width}px solid var(--{p}-color-action-primary-bg);
  outline-offset: {BORDERS.focus_ring_offset}px;
}}
"""


def style_tag() -> str:
    """The stylesheet wrapped in a ``<style>`` element. Contains only kit-generated text."""
    return f'<style id="{STYLE_ELEMENT_ID}">\n{build_css()}</style>'


def inject_global_styles() -> None:
    """Add the global stylesheet to the current Streamlit run.

    Call once per script run from the app shell, right after ``st.set_page_config``.
    Streamlit rebuilds the page on every rerun, so the shell must call this on each run;
    calling it more than once in a run only repeats an identical stylesheet, which has no
    visual effect. The content is generated entirely from tokens: no user input and no
    script is ever inserted. Streamlit is imported here, not at module level, so tokens and
    theme stay importable without it.
    """
    import streamlit as st

    st.html(style_tag())
