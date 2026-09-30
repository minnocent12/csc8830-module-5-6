"""Safe image IO for Module 5-6.

OpenCV decodes color images in BGR order. This foundation module preserves that convention
and leaves conversion to RGB for display code. It underlies both the four-view Structure
from Motion images and individual video frames extracted in a later phase.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

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


_EXIF_ORIENTATION_TAG = 0x0112


def read_exif_orientation(path: str | Path) -> int:
    """Read the EXIF ``Orientation`` tag (1-8) from an image file.

    Returns ``1`` (meaning "no rotation needed") both when the image genuinely has no
    rotation and when it has no EXIF orientation tag at all (e.g. a screenshot or a
    non-photo image) - both cases require no correction.
    """
    try:
        with Image.open(path) as img:
            orientation = img.getexif().get(_EXIF_ORIENTATION_TAG)
    except (OSError, UnidentifiedImageError):
        return 1
    return int(orientation) if orientation else 1


def apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate/flip an image array to correct for a standard EXIF ``Orientation`` tag (1-8).

    Never mutates the input; always returns a new array. ``orientation=1`` or any value
    outside ``1..8`` returns an unrotated copy. See the EXIF specification's Orientation tag
    (0x0112) for the eight standard values.
    """
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def load_image_bgr_oriented(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a photo as a BGR image, with its EXIF ``Orientation`` tag applied exactly once.

    Returns ``(oriented_image, orientation_used)``. Real phone photos are very commonly
    stored using the sensor's native pixel layout (often landscape, even for a photo the
    user took in portrait) plus an EXIF Orientation tag describing how a viewer should rotate
    them for correct display - important for anything (like manually identified boundary
    corners) that depends on pixel coordinates agreeing with the displayed image.

    This loads the raw, un-rotated sensor pixels explicitly (``cv2.IMREAD_IGNORE_ORIENTATION``)
    and then applies the correction ourselves, rather than relying on plain ``cv2.imread``'s own
    built-in EXIF handling: whether that default applies the rotation automatically differs by
    OpenCV build (observed to auto-rotate on OpenCV 5.0.0 here, but this is not guaranteed on
    every version), so depending on it implicitly would make loaded pixel coordinates
    non-reproducible across environments. Applying the rotation explicitly, exactly once, keeps
    this deterministic regardless of the OpenCV build running it.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if raw is None:
        raise ValueError(f"could not decode color image: {image_path}")
    raw = validate_image_array(raw, name="loaded BGR image", color_order="BGR").copy()
    orientation = read_exif_orientation(path)
    return apply_exif_orientation(raw, orientation), orientation


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
