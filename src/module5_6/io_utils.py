"""Safe image IO for Module 5-6.

OpenCV decodes color images in BGR order. This foundation module preserves that convention
and leaves conversion to RGB for display code. It underlies both the four-view Structure
from Motion images and individual video frames extracted in a later phase.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from module5_6.types import ColorOrder, ImageMetadata

SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


def validate_image_array(
    image: np.ndarray,
    *,
    name: str = "image",
    color_order: ColorOrder | None = None,
) -> np.ndarray:
    """Validate a non-empty 2D or 3D image array and return it unchanged.

    The function never mutates the input. Channel counts are restricted to common OpenCV
    grayscale, BGR/RGB, and four-channel image representations.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D or 3D, got {image.ndim}D")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{name} must have positive height and width")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels, got {image.shape[2]}")
    if image.dtype.kind in "fc" and not np.isfinite(image).all():
        raise ValueError(f"{name} contains non-finite values")
    if color_order == "GRAY" and image.ndim == 3 and image.shape[2] != 1:
        raise ValueError("GRAY images must be 2D or have one channel")
    if color_order in ("BGR", "RGB") and image.ndim == 2:
        raise ValueError(f"{color_order} images must have color channels")
    return image


def decode_image_bgr(data: bytes, *, source_name: str | None = None) -> np.ndarray:
    """Decode uploaded bytes as a copied OpenCV BGR image."""
    if not data:
        raise ValueError("image data is empty")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        label = f" from {source_name}" if source_name else ""
        raise ValueError(f"could not decode color image{label}")
    return validate_image_array(image, name="decoded BGR image", color_order="BGR").copy()


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load a path as a BGR uint8 image without modifying the source file."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode color image: {image_path}")
    return validate_image_array(image, name="loaded BGR image", color_order="BGR").copy()


def load_image_unchanged(path: str | Path) -> np.ndarray:
    """Load an image while preserving its source bit depth and channel count."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode image: {image_path}")
    return validate_image_array(image, name="loaded unchanged image").copy()


def to_grayscale(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    """Convert a validated BGR or grayscale image to single-channel grayscale.

    Returns the validated input unchanged (not a copy) when it is already 2D grayscale, so
    callers that already hold a grayscale frame avoid a redundant conversion.
    """
    validated = validate_image_array(image, name=name)
    if validated.ndim == 2:
        return validated
    if validated.shape[2] != 3:
        raise ValueError(f"{name} must be single-channel grayscale or 3-channel BGR, got {validated.shape[2]} channels")
    return cv2.cvtColor(validated, cv2.COLOR_BGR2GRAY)


def describe_image(
    image: np.ndarray,
    *,
    color_order: ColorOrder,
    source_name: str | None = None,
) -> ImageMetadata:
    """Return shape/dtype metadata after validating the image."""
    validated = validate_image_array(image, color_order=color_order)
    channels = 1 if validated.ndim == 2 else validated.shape[2]
    return ImageMetadata(
        height=validated.shape[0],
        width=validated.shape[1],
        channels=channels,
        dtype=str(validated.dtype),
        color_order=color_order,
        source_name=source_name,
    )
