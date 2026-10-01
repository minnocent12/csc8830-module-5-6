"""Design tokens for the CSc 8830 Computer Vision application.

This module is framework independent: it must never import Streamlit or any UI library, so
it can be tested on its own and later turned into CSS or a Streamlit theme. Internal imports
are relative so the package can be vendored under any parent package name.

Token groups:
    COLORS        semantic color roles (brand, actions, neutrals, feedback states)
    TYPOGRAPHY    text roles (size, weight, line height, tracking, case, font stack)
    HEADINGS      h1 to h6 sizes and weights, main area and sidebar
    LAYOUT        content max widths
    SPACING       4px-based spacing scale
    RADII         corner radii
    BORDERS       border widths and colors, focus ring treatment
    ELEVATION     shadows (deliberately minimal)
    BREAKPOINTS   viewport widths for later responsive work
    CONTRAST_PAIRS  every foreground/background pairing the kit relies on, with its WCAG use

Units: every length is in CSS pixels (int). Colors are opaque sRGB ``#RRGGBB`` strings, except
shadow values, which are complete CSS ``box-shadow`` strings.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from .accessibility import ContrastUse

# Raw palette. Private on purpose: application code should use the semantic roles in COLORS,
# so a role can be retuned without hunting for hex values.
_ORANGE_500 = "#F96302"  # brand orange
_ORANGE_50 = "#FFF1E8"
_ORANGE_700 = "#C2410C"  # darkest orange that is not brown; white text passes AA on it
_ORANGE_800 = "#9A3412"
_ORANGE_900 = "#7C2D12"
_WHITE = "#FFFFFF"
_GRAY_50 = "#F6F6F4"
_GRAY_100 = "#EFEFEC"
_GRAY_300 = "#D6D6D1"
_GRAY_500 = "#8A8A84"
_GRAY_550 = "#8C8C86"
_GRAY_700 = "#5C5C57"
_GRAY_850 = "#2E2E2B"
_GRAY_950 = "#1A1A1A"
_GREEN_700 = "#166534"
_GREEN_50 = "#EAF6EE"
_AMBER_800 = "#8A4B00"
_AMBER_50 = "#FDF4E3"
_RED_700 = "#B42318"
_RED_50 = "#FDEDEC"
_BLUE_700 = "#1F4FBF"
_BLUE_50 = "#EAF0FC"


@dataclass(frozen=True)
class ColorTokens:
    """Semantic color roles.

    Usage rules that the contrast tests enforce:
    * ``brand_orange`` is too light for white text (3.08:1) and for state indicators on the
      app background (2.85:1). Use it for fills that carry ``text_on_brand``, for accents on
      white surfaces, and for decoration.
    * Primary actions, focus rings, and active borders use ``action_primary_bg`` instead.
    * ``border`` is decorative only; input outlines must use ``border_strong``.
    """

    # Brand
    brand_orange: str = _ORANGE_500
    brand_orange_soft: str = _ORANGE_50
    brand_orange_strong: str = _ORANGE_800
    text_on_brand: str = _GRAY_950

    # Actions
    action_primary_bg: str = _ORANGE_700
    action_primary_fg: str = _WHITE
    action_primary_hover: str = _ORANGE_800
    action_primary_pressed: str = _ORANGE_900

    # Neutrals
    app_background: str = _GRAY_50
    surface_primary: str = _WHITE
    surface_secondary: str = _GRAY_100
    border: str = _GRAY_300
    border_strong: str = _GRAY_500
    text_heading: str = _GRAY_950
    text_body: str = _GRAY_850
    text_muted: str = _GRAY_700
    text_disabled: str = _GRAY_550

    # Feedback states. Each foreground passes normal-text contrast on its soft background and
    # on white, so chips and banners never depend on color alone being legible.
    success: str = _GREEN_700
    success_soft: str = _GREEN_50
    warning: str = _AMBER_800
    warning_soft: str = _AMBER_50
    error: str = _RED_700
    error_soft: str = _RED_50
    info: str = _BLUE_700
    info_soft: str = _BLUE_50


@dataclass(frozen=True)
class TypeRole:
    """One typographic role. ``line_height`` is unitless; ``letter_spacing_em`` is in em."""

    size_px: int
    weight: int
    line_height: float
    letter_spacing_em: float = 0.0
    uppercase: bool = False
    monospace: bool = False


# Native system stacks: no web font dependency in this phase.
FONT_FAMILY_SANS = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, '
    'sans-serif, "Apple Color Emoji", "Segoe UI Emoji"'
)
FONT_FAMILY_MONO = (
    'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace'
)


@dataclass(frozen=True)
class TypographyTokens:
    """Text roles shared by experiment pages and theory pages alike."""

    app_title: TypeRole = TypeRole(size_px=18, weight=700, line_height=1.3)
    module_eyebrow: TypeRole = TypeRole(
        size_px=12, weight=700, line_height=1.4, letter_spacing_em=0.08, uppercase=True
    )
    page_title: TypeRole = TypeRole(
        size_px=32, weight=700, line_height=1.2, letter_spacing_em=-0.01
    )
    section_title: TypeRole = TypeRole(size_px=22, weight=600, line_height=1.3)
    subsection_title: TypeRole = TypeRole(size_px=18, weight=600, line_height=1.4)
    body: TypeRole = TypeRole(size_px=16, weight=400, line_height=1.6)
    body_small: TypeRole = TypeRole(size_px=14, weight=400, line_height=1.5)
    caption: TypeRole = TypeRole(size_px=13, weight=400, line_height=1.45)
    # Not uppercase: metric labels carry units, and case is meaningful (dB vs DB, mm vs MM).
    metric_label: TypeRole = TypeRole(
        size_px=13, weight=600, line_height=1.4, letter_spacing_em=0.02
    )
    metric_value: TypeRole = TypeRole(size_px=28, weight=700, line_height=1.15)
    code: TypeRole = TypeRole(size_px=14, weight=400, line_height=1.5, monospace=True)


@dataclass(frozen=True)
class HeadingTokens:
    """Sizes (px) and weights for HTML heading levels h1 to h6, index 0 being h1.

    Streamlit themes headings by tag, so these bind the typography roles to tags:
    h1 = page_title, h3 = section_title, h4 = subsection_title. Current pages use
    ``st.header`` (h2) for their title and ``st.subheader`` (h3) for sections; h2 sits between
    so long-form documents that nest their own h1 still read in order. Weights are multiples
    of 100 because Streamlit rejects other values. The sidebar has its own compact scale, with
    h1 as the app title.
    """

    sizes_px: tuple[int, ...] = (32, 26, 22, 18, 16, 14)
    weights: tuple[int, ...] = (700, 700, 600, 600, 600, 600)
    sidebar_sizes_px: tuple[int, ...] = (18, 16, 15, 14, 13, 12)
    sidebar_weights: tuple[int, ...] = (700, 600, 600, 600, 600, 600)


@dataclass(frozen=True)
class LayoutTokens:
    """Content widths in px. Pages use Streamlit's wide layout; these cap line length."""

    content_max_width: int = 1360  # experiment and visualization pages
    reading_max_width: int = 760  # long-form theory text, applied in a later phase


@dataclass(frozen=True)
class SpacingTokens:
    """Spacing scale on a 4px base unit. Field ``xN`` equals N base units."""

    x1: int = 4
    x2: int = 8
    x3: int = 12
    x4: int = 16
    x6: int = 24
    x8: int = 32
    x12: int = 48
    x16: int = 64


SPACING_BASE_UNIT = 4


@dataclass(frozen=True)
class RadiusTokens:
    """Corner radii. Kept small: structure comes from borders and spacing, not rounding."""

    small: int = 2  # inputs, inline code, table cells
    standard: int = 4  # buttons, tabs, banners
    card: int = 6  # cards, image frames, configuration panels
    pill: int = 999  # status chips; large enough to fully round any chip height


@dataclass(frozen=True)
class BorderTokens:
    """Border widths and colors, plus the keyboard focus ring.

    The focus ring is drawn outside the element (``focus_ring_offset`` px gap) so it stays
    visible on both white and app-background surfaces.
    """

    width_default: int = 1
    width_strong: int = 1
    width_active: int = 2
    color_default: str = ColorTokens.border
    color_strong: str = ColorTokens.border_strong
    color_active: str = ColorTokens.action_primary_bg
    focus_ring_color: str = ColorTokens.action_primary_bg
    focus_ring_width: int = 2
    focus_ring_offset: int = 2


@dataclass(frozen=True)
class ElevationTokens:
    """Shadows as CSS ``box-shadow`` values. Cards rely on borders first."""

    none: str = "none"
    card: str = "0 1px 2px rgba(26, 26, 26, 0.06)"
    raised: str = "0 2px 6px rgba(26, 26, 26, 0.08)"


@dataclass(frozen=True)
class BreakpointTokens:
    """Minimum viewport widths (px) at which each layout tier begins.

    ``compact`` matches the width below which Streamlit already stacks ``st.columns``.
    Responsive rules that use these values arrive in a later phase.
    """

    compact: int = 640  # below this: phones, single column
    medium: int = 900  # tablets and narrow laptop windows: two columns at most
    wide: int = 1200  # normal laptops: up to three columns
    extra_wide: int = 1600  # large desktops: four column galleries allowed


COLORS = ColorTokens()
TYPOGRAPHY = TypographyTokens()
HEADINGS = HeadingTokens()
LAYOUT = LayoutTokens()
SPACING = SpacingTokens()
RADII = RadiusTokens()
BORDERS = BorderTokens()
ELEVATION = ElevationTokens()
BREAKPOINTS = BreakpointTokens()


@dataclass(frozen=True)
class ContrastPair:
    """A foreground/background combination the kit relies on, and what it is used for."""

    foreground: str  # ColorTokens field name
    background: str  # ColorTokens field name
    use: ContrastUse
    note: str = ""


CONTRAST_PAIRS: tuple[ContrastPair, ...] = (
    # Actions
    ContrastPair("action_primary_fg", "action_primary_bg", ContrastUse.NORMAL_TEXT, "primary CTA label"),
    ContrastPair("action_primary_fg", "action_primary_hover", ContrastUse.NORMAL_TEXT, "CTA hover"),
    ContrastPair("action_primary_fg", "action_primary_pressed", ContrastUse.NORMAL_TEXT, "CTA pressed"),
    # Text on the bright brand surface
    ContrastPair("text_on_brand", "brand_orange", ContrastUse.NORMAL_TEXT, "dark text on brand fill"),
    ContrastPair("text_heading", "brand_orange_soft", ContrastUse.NORMAL_TEXT, "selected item label"),
    ContrastPair("brand_orange_strong", "brand_orange_soft", ContrastUse.NORMAL_TEXT, "orange text on soft"),
    ContrastPair("brand_orange_strong", "surface_primary", ContrastUse.NORMAL_TEXT, "orange link text"),
    ContrastPair("brand_orange_strong", "app_background", ContrastUse.NORMAL_TEXT, "eyebrow, links"),
    # Body text on every neutral surface
    ContrastPair("text_heading", "app_background", ContrastUse.NORMAL_TEXT),
    ContrastPair("text_body", "app_background", ContrastUse.NORMAL_TEXT),
    ContrastPair("text_body", "surface_primary", ContrastUse.NORMAL_TEXT),
    ContrastPair("text_body", "surface_secondary", ContrastUse.NORMAL_TEXT),
    ContrastPair("text_muted", "app_background", ContrastUse.NORMAL_TEXT, "captions"),
    ContrastPair("text_muted", "surface_primary", ContrastUse.NORMAL_TEXT, "captions"),
    ContrastPair("text_muted", "surface_secondary", ContrastUse.NORMAL_TEXT, "captions"),
    # Disabled text is exempt under SC 1.4.3; it is held to 3:1 so it stays readable.
    ContrastPair("text_disabled", "surface_primary", ContrastUse.LARGE_TEXT, "disabled, exempt"),
    # Non-text indicators
    ContrastPair("action_primary_bg", "app_background", ContrastUse.NON_TEXT, "focus ring, active border"),
    ContrastPair("action_primary_bg", "surface_primary", ContrastUse.NON_TEXT, "focus ring, active border"),
    ContrastPair("action_primary_bg", "surface_secondary", ContrastUse.NON_TEXT, "focus ring, active border"),
    ContrastPair("border_strong", "surface_primary", ContrastUse.NON_TEXT, "input outline"),
    ContrastPair("border_strong", "app_background", ContrastUse.NON_TEXT, "input outline"),
    ContrastPair("brand_orange", "surface_primary", ContrastUse.NON_TEXT, "accent on white only"),
    # Brand orange on the app background is 2.85:1: decoration only, never a sole indicator.
    ContrastPair("brand_orange", "app_background", ContrastUse.DECORATIVE, "decoration only"),
    ContrastPair("border", "surface_primary", ContrastUse.DECORATIVE, "card separation"),
    # Feedback states on their soft backgrounds and on white
    ContrastPair("success", "success_soft", ContrastUse.NORMAL_TEXT),
    ContrastPair("success", "surface_primary", ContrastUse.NORMAL_TEXT),
    ContrastPair("warning", "warning_soft", ContrastUse.NORMAL_TEXT),
    ContrastPair("warning", "surface_primary", ContrastUse.NORMAL_TEXT),
    ContrastPair("error", "error_soft", ContrastUse.NORMAL_TEXT),
    ContrastPair("error", "surface_primary", ContrastUse.NORMAL_TEXT),
    ContrastPair("info", "info_soft", ContrastUse.NORMAL_TEXT),
    ContrastPair("info", "surface_primary", ContrastUse.NORMAL_TEXT),
    ContrastPair("text_body", "success_soft", ContrastUse.NORMAL_TEXT, "banner body text"),
    ContrastPair("text_body", "warning_soft", ContrastUse.NORMAL_TEXT, "banner body text"),
    ContrastPair("text_body", "error_soft", ContrastUse.NORMAL_TEXT, "banner body text"),
    ContrastPair("text_body", "info_soft", ContrastUse.NORMAL_TEXT, "banner body text"),
)


def color_roles() -> dict[str, str]:
    """All semantic color roles as ``{role_name: "#RRGGBB"}``."""
    return {f.name: getattr(COLORS, f.name) for f in fields(ColorTokens)}
