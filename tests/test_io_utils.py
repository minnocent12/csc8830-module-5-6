from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from module5_6.io_utils import (
    apply_exif_orientation,
    decode_image_bgr,
    describe_image,
    load_image_bgr_oriented,
    read_exif_orientation,
    to_grayscale,
    validate_image_array,
)


def test_validate_image_array_preserves_pixels() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[1, 2] = (1, 2, 3)
    original = image.copy()

    result = validate_image_array(image, color_order="BGR")

    assert result is image
    np.testing.assert_array_equal(image, original)


def test_validate_image_array_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        validate_image_array(np.zeros((2, 3, 4, 1), dtype=np.uint8))


def test_decode_image_bgr_returns_expected_shape_and_order() -> None:
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    source[1, 2] = (10, 20, 30)
    success, encoded = cv2.imencode(".png", source)
    assert success

    decoded = decode_image_bgr(encoded.tobytes())

    assert decoded.shape == source.shape
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, source)


def test_decode_image_bgr_rejects_empty_bytes() -> None:
    with pytest.raises(ValueError, match="empty"):
        decode_image_bgr(b"")


def test_to_grayscale_converts_bgr_image() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., 2] = 255  # pure red in BGR

    gray = to_grayscale(image)

    assert gray.ndim == 2
    assert gray.shape == (4, 5)
    assert gray.dtype == np.uint8


def test_to_grayscale_passes_through_existing_grayscale() -> None:
    gray_source = np.arange(12, dtype=np.uint8).reshape(3, 4)

    result = to_grayscale(gray_source)

    assert result is gray_source


def test_to_grayscale_rejects_four_channel_image() -> None:
    with pytest.raises(ValueError, match="single-channel grayscale or 3-channel BGR"):
        to_grayscale(np.zeros((4, 4, 4), dtype=np.uint8))


def test_describe_image_records_shape_dtype_and_channel_order() -> None:
    metadata = describe_image(
        np.zeros((7, 8, 3), dtype=np.uint8),
        color_order="BGR",
        source_name="fixture.png",
    )

    assert metadata.height == 7
    assert metadata.width == 8
    assert metadata.channels == 3
    assert metadata.dtype == "uint8"
    assert metadata.color_order == "BGR"
    assert metadata.source_name == "fixture.png"


def _marker_image() -> np.ndarray:
    """A 3-row x 4-col BGR image with a single white marker pixel at (row=0, col=0)."""
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[0, 0] = (255, 255, 255)
    return image


# For each EXIF orientation code, the (row, col) the top-left marker pixel lands at, and the
# resulting (height, width) - expected values per the EXIF Orientation tag (0x0112) spec,
# confirmed against this repository's cv2-based implementation.
_ORIENTATION_EXPECTATIONS = {
    1: ((0, 0), (3, 4)),
    2: ((0, 3), (3, 4)),
    3: ((2, 3), (3, 4)),
    4: ((2, 0), (3, 4)),
    5: ((0, 0), (4, 3)),
    6: ((0, 2), (4, 3)),
    7: ((3, 2), (4, 3)),
    8: ((3, 0), (4, 3)),
}


@pytest.mark.parametrize("orientation", sorted(_ORIENTATION_EXPECTATIONS))
def test_apply_exif_orientation_moves_marker_as_expected(orientation: int) -> None:
    image = _marker_image()
    (expected_row, expected_col), expected_shape = _ORIENTATION_EXPECTATIONS[orientation]

    result = apply_exif_orientation(image, orientation)

    assert result.shape == (*expected_shape, 3)
    rows, cols = np.where(result[..., 0] == 255)
    assert (rows[0], cols[0]) == (expected_row, expected_col)


def test_apply_exif_orientation_does_not_mutate_input() -> None:
    image = _marker_image()
    original = image.copy()

    apply_exif_orientation(image, 6)

    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("orientation", [0, 9, -1])
def test_apply_exif_orientation_treats_unknown_codes_as_no_rotation(orientation: int) -> None:
    image = _marker_image()

    result = apply_exif_orientation(image, orientation)

    assert result.shape == image.shape
    np.testing.assert_array_equal(result, image)
    assert result is not image


def _save_jpeg_with_orientation(path: Path, orientation: int | None) -> None:
    image = Image.new("RGB", (4, 3), color=(255, 0, 0))
    if orientation is None:
        image.save(path, format="JPEG")
        return
    exif = image.getexif()
    exif[0x0112] = orientation
    image.save(path, format="JPEG", exif=exif)


def test_read_exif_orientation_reads_actual_tag(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "rotated.jpg"
    _save_jpeg_with_orientation(jpeg_path, 6)

    assert read_exif_orientation(jpeg_path) == 6


def test_read_exif_orientation_defaults_to_one_without_exif(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "plain.jpg"
    _save_jpeg_with_orientation(jpeg_path, None)

    assert read_exif_orientation(jpeg_path) == 1


def test_read_exif_orientation_defaults_to_one_for_non_photo_png(tmp_path: Path) -> None:
    png_path = tmp_path / "plain.png"
    cv2.imwrite(str(png_path), np.zeros((3, 4, 3), dtype=np.uint8))

    assert read_exif_orientation(png_path) == 1


def test_load_image_bgr_oriented_rotates_and_reports_orientation(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "rotated.jpg"
    # 6x4 (h x w) source with a distinct marker pixel so the rotation is verifiable.
    source = np.zeros((6, 4, 3), dtype=np.uint8)
    source[0, 0] = (0, 0, 255)  # BGR red, at the top-left corner
    success, encoded = cv2.imencode(".jpg", source)
    assert success
    pil_image = Image.open(io.BytesIO(encoded.tobytes()))
    pil_image.load()
    exif = pil_image.getexif()
    exif[0x0112] = 6
    pil_image.save(jpeg_path, format="JPEG", exif=exif)

    oriented, orientation_used = load_image_bgr_oriented(jpeg_path)

    assert orientation_used == 6
    assert oriented.shape == (4, 6, 3)  # height/width swapped by the 90-degree rotation
    # JPEG is lossy, so match the marker by nearest-red rather than exact equality.
    red_channel = oriented[..., 2].astype(int)
    row, col = np.unravel_index(np.argmax(red_channel), red_channel.shape)
    assert (row, col) == (0, 5)  # matches the orientation=6 marker mapping verified against cv2 directly


def test_load_image_bgr_oriented_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image_bgr_oriented(tmp_path / "does_not_exist.jpg")
