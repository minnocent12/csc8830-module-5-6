"""Module 5-6 Streamlit pages and the get_pages provider.

The foundation phase exposes the final five-page navigation shape with honest pending
notices. The optical-flow, tracking, bilinear-interpolation, and structure-from-motion
implementations are intentionally added in later approved phases (see IMPLEMENTATION_PLAN.md).
"""
from __future__ import annotations

from module5_6.webapp._page import PageSpec
from module5_6.webapp.ui import foundation_page

_MODULE = "Module 5-6"


def _optical_flow_page() -> None:
    foundation_page(
        "Optical Flow",
        "Question 1: loading a 30-second sample from a user video, computing dense "
        "(Farneback) and/or sparse (Lucas-Kanade) optical flow, and visualizing the result as "
        "a video will be added in a later phase.",
    )


def _motion_tracking_page() -> None:
    foundation_page(
        "Motion Tracking",
        "Question 1: Shi-Tomasi feature detection, pyramidal Lucas-Kanade tracking between two "
        "frames, and the two-frame tracking-problem visualization will be added in a later "
        "phase.",
    )


def _theory_page() -> None:
    foundation_page(
        "Bilinear Interpolation & Theory",
        "Question 1: the brightness-constancy derivation, the optical-flow constraint "
        "equation, the aperture problem, the Lucas-Kanade least-squares derivation, and an "
        "interactive bilinear-interpolation demonstration will be added in a later phase.",
    )


def _sfm_page() -> None:
    foundation_page(
        "Structure From Motion",
        "Question 2: four-viewpoint image upload, feature correspondence, planar homography "
        "estimation, reprojection validation, and recovered-boundary visualization will be "
        "added in a later phase.",
    )


def _experiments_page() -> None:
    foundation_page(
        "Experiments & Results",
        "Consolidated video summaries, tracking-error tables, four-view SfM experiment "
        "results, camera information, and reprojection results will be added once the real "
        "videos and four-view images have been processed in later phases.",
    )


def get_pages() -> list[PageSpec]:
    """Return the Module 5-6 pages for standalone or shared-dashboard hosts."""
    return [
        PageSpec(_MODULE, "Optical Flow", 10, _optical_flow_page),
        PageSpec(_MODULE, "Motion Tracking", 20, _motion_tracking_page),
        PageSpec(_MODULE, "Bilinear Interpolation & Theory", 30, _theory_page),
        PageSpec(_MODULE, "Structure From Motion", 40, _sfm_page),
        PageSpec(_MODULE, "Experiments & Results", 50, _experiments_page),
    ]
