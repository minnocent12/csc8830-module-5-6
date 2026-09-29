from __future__ import annotations

import cv2
import numpy as np
import pytest

from module5_6.io_utils import (
    decode_image_bgr,
    describe_image,
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
