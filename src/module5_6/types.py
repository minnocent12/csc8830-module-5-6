"""Small, dependency-light types shared by Module 5-6 foundation code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColorOrder = Literal["BGR", "RGB", "GRAY"]
ExperimentStatusValue = Literal["pending", "completed", "failed", "unavailable"]


@dataclass(frozen=True)
class ImageMetadata:
    """Describes an image or video-frame array without changing or copying its pixels."""

    height: int
    width: int
    channels: int
    dtype: str
    color_order: ColorOrder
    source_name: str | None = None


@dataclass(frozen=True)
class ExperimentStatus:
    """Reports whether a required real-data experiment (video or four-view SfM) has run.

    Used to keep optical-flow tracking validation and structure-from-motion reprojection
    results honestly labeled until the corresponding real videos or images are processed.
    """

    status: ExperimentStatusValue
    message: str
    source: str | None = None
