from __future__ import annotations

import cv2
import numpy as np
import pytest

from module5_6.interpolation import (
    bilinear_interpolate,
    bilinear_interpolate_corners,
    bilinear_weights,
)


def test_bilinear_weights_reduce_to_single_corner_at_grid_points() -> None:
    assert bilinear_weights(0.0, 0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert bilinear_weights(1.0, 0.0) == pytest.approx((0.0, 1.0, 0.0, 0.0))
    assert bilinear_weights(0.0, 1.0) == pytest.approx((0.0, 0.0, 1.0, 0.0))
    assert bilinear_weights(1.0, 1.0) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_bilinear_weights_sum_to_one() -> None:
    for alpha, beta in [(0.3, 0.7), (0.5, 0.5), (0.1, 0.9), (0.99, 0.01)]:
        weights = bilinear_weights(alpha, beta)
        assert sum(weights) == pytest.approx(1.0)


def test_bilinear_weights_rejects_out_of_range_alpha_or_beta() -> None:
    with pytest.raises(ValueError, match="alpha"):
        bilinear_weights(1.5, 0.5)
    with pytest.raises(ValueError, match="beta"):
        bilinear_weights(0.5, -0.1)


def test_bilinear_interpolate_corners_matches_manual_calculation() -> None:
    """I00=10, I10=20, I01=30, I11=40 at alpha=0.3, beta=0.7 is a hand-worked example.

    Manual calculation: w00=0.21, w10=0.09, w01=0.49, w11=0.21;
    value = 0.21*10 + 0.09*20 + 0.49*30 + 0.21*40 = 27.0.
    This is a mathematical worked example, not assignment experimental data.
    """
    value = bilinear_interpolate_corners(i00=10, i10=20, i01=30, i11=40, alpha=0.3, beta=0.7)

    assert value == pytest.approx(27.0)


def test_bilinear_interpolate_corners_center_is_the_average_of_four_corners() -> None:
    value = bilinear_interpolate_corners(i00=10, i10=20, i01=30, i11=40, alpha=0.5, beta=0.5)

    assert value == pytest.approx((10 + 20 + 30 + 40) / 4.0)


def test_bilinear_interpolate_at_integer_coordinate_returns_exact_pixel() -> None:
    image = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)

    sample = bilinear_interpolate(image, x=1.0, y=1.0)

    assert sample.value == pytest.approx(50.0)
    assert sample.alpha == pytest.approx(0.0)
    assert sample.beta == pytest.approx(0.0)


def test_bilinear_interpolate_at_center_of_four_known_pixels() -> None:
    image = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

    sample = bilinear_interpolate(image, x=0.5, y=0.5)

    assert sample.i00 == pytest.approx(10.0)
    assert sample.i10 == pytest.approx(20.0)
    assert sample.i01 == pytest.approx(30.0)
    assert sample.i11 == pytest.approx(40.0)
    assert sample.value == pytest.approx(25.0)


def test_bilinear_interpolate_arbitrary_fractional_coordinate() -> None:
    image = np.array(
        [[0.0, 10.0, 20.0], [10.0, 20.0, 30.0], [20.0, 30.0, 40.0]], dtype=np.float64
    )

    sample = bilinear_interpolate(image, x=1.3, y=0.7)

    # Manual check: x0=1, y0=0, x1=2, y1=1, alpha=0.3, beta=0.7
    # I00=I(1,0)=10, I10=I(2,0)=20, I01=I(1,1)=20, I11=I(2,1)=30
    expected = bilinear_interpolate_corners(i00=10.0, i10=20.0, i01=20.0, i11=30.0, alpha=0.3, beta=0.7)
    assert sample.value == pytest.approx(expected)
    assert sample.x0 == 1 and sample.y0 == 0 and sample.x1 == 2 and sample.y1 == 1


def test_bilinear_interpolate_weights_and_breakdown_are_self_consistent() -> None:
    image = np.arange(16, dtype=np.float64).reshape(4, 4)

    sample = bilinear_interpolate(image, x=2.25, y=1.75)

    assert sum([sample.weight_00, sample.weight_10, sample.weight_01, sample.weight_11]) == pytest.approx(1.0)
    manual = (
        sample.weight_00 * sample.i00
        + sample.weight_10 * sample.i10
        + sample.weight_01 * sample.i01
        + sample.weight_11 * sample.i11
    )
    assert sample.value == pytest.approx(manual)


def test_bilinear_interpolate_rejects_coordinate_outside_valid_range() -> None:
    image = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="outside the valid interpolation range"):
        bilinear_interpolate(image, x=-0.1, y=0.0)
    with pytest.raises(ValueError, match="outside the valid interpolation range"):
        bilinear_interpolate(image, x=0.0, y=3.5)


def test_bilinear_interpolate_rejects_too_small_image() -> None:
    with pytest.raises(ValueError, match="at least 2x2"):
        bilinear_interpolate(np.zeros((1, 5), dtype=np.float64), x=0.0, y=0.0)


def test_bilinear_interpolate_rejects_non_2d_image() -> None:
    with pytest.raises(ValueError, match="single-channel"):
        bilinear_interpolate(np.zeros((4, 4, 3), dtype=np.float64), x=0.0, y=0.0)


def test_bilinear_interpolate_rejects_non_array_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        bilinear_interpolate([[1, 2], [3, 4]], x=0.0, y=0.0)


def test_bilinear_interpolate_agrees_with_opencv_remap_reference() -> None:
    """Cross-validation against cv2.remap(INTER_LINEAR) - software verification only."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(20, 20)).astype(np.float32)

    for x, y in [(5.25, 5.75), (10.5, 10.5), (3.1, 15.9), (0.5, 0.5), (18.5, 2.25)]:
        ours = bilinear_interpolate(image, x, y).value
        map_x = np.array([[x]], dtype=np.float32)
        map_y = np.array([[y]], dtype=np.float32)
        reference = float(
            cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)[0, 0]
        )
        assert ours == pytest.approx(reference, abs=1e-2)
