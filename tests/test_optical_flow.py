from __future__ import annotations

import cv2
import numpy as np
import pytest

from module5_6.optical_flow import (
    FarnebackParams,
    compute_farneback_flow,
    draw_flow_arrows,
    flow_magnitude_angle,
    flow_to_hsv_bgr,
    summarize_flow_magnitudes,
)


def _textured_frame(size: tuple[int, int] = (140, 140)) -> np.ndarray:
    """A deterministic textured grayscale frame; Farneback needs gradients to estimate flow."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=size, dtype=np.uint8)


def _translated(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate frame by a known (dx, dy) pixel shift, replicating border pixels."""
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    height, width = frame.shape[:2]
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def test_compute_farneback_flow_recovers_known_translation() -> None:
    """A known synthetic (dx, dy) = (5, 3) shift is software verification, not assignment data."""
    dx, dy = 5, 3
    previous = _textured_frame()
    following = _translated(previous, dx, dy)

    flow = compute_farneback_flow(previous, following)

    assert flow.shape == (*previous.shape, 2)
    assert flow.dtype == np.float32
    margin = 20
    interior = flow[margin:-margin, margin:-margin]
    mean_u = float(interior[..., 0].mean())
    mean_v = float(interior[..., 1].mean())
    assert abs(mean_u - dx) < 1.0
    assert abs(mean_v - dy) < 1.0


def test_compute_farneback_flow_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_farneback_flow(np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 11), dtype=np.uint8))


def test_compute_farneback_flow_rejects_color_input() -> None:
    with pytest.raises(ValueError, match="single-channel"):
        compute_farneback_flow(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8))


def test_flow_magnitude_angle_matches_manual_vector_math() -> None:
    flow = np.zeros((2, 2, 2), dtype=np.float32)
    flow[0, 0] = (3.0, 4.0)  # magnitude 5, angle 53.13 degrees
    flow[0, 1] = (0.0, 0.0)

    magnitude, angle_degrees = flow_magnitude_angle(flow)

    assert magnitude[0, 0] == pytest.approx(5.0, abs=1e-4)
    assert angle_degrees[0, 0] == pytest.approx(53.13, abs=0.1)
    assert magnitude[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_flow_magnitude_angle_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="H, W, 2"):
        flow_magnitude_angle(np.zeros((4, 4, 3), dtype=np.float32))


def test_flow_to_hsv_bgr_returns_valid_display_image() -> None:
    flow = np.zeros((8, 8, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    flow[..., 1] = 0.0

    visualization = flow_to_hsv_bgr(flow)

    assert visualization.shape == (8, 8, 3)
    assert visualization.dtype == np.uint8


def test_flow_to_hsv_bgr_handles_all_zero_flow() -> None:
    visualization = flow_to_hsv_bgr(np.zeros((4, 4, 2), dtype=np.float32))
    assert visualization.shape == (4, 4, 3)
    np.testing.assert_array_equal(visualization, 0)


def test_draw_flow_arrows_does_not_mutate_input_frame() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    original = frame.copy()
    flow = np.full((32, 32, 2), 5.0, dtype=np.float32)

    overlay = draw_flow_arrows(frame, flow, step=8)

    np.testing.assert_array_equal(frame, original)
    assert overlay.shape == frame.shape
    assert not np.array_equal(overlay, frame)


def test_draw_flow_arrows_skips_motion_below_threshold() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    flow = np.full((16, 16, 2), 0.1, dtype=np.float32)

    overlay = draw_flow_arrows(frame, flow, step=8, min_magnitude=1.0)

    np.testing.assert_array_equal(overlay, frame)


def test_draw_flow_arrows_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same height and width"):
        draw_flow_arrows(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((5, 5, 2), dtype=np.float32))


def test_summarize_flow_magnitudes_computes_expected_statistics() -> None:
    magnitudes = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])]

    stats = summarize_flow_magnitudes(magnitudes)

    assert stats.frame_pairs == 2
    assert stats.mean_magnitude == pytest.approx(4.5)
    assert stats.median_magnitude == pytest.approx(4.5)
    assert stats.max_magnitude == pytest.approx(8.0)


def test_summarize_flow_magnitudes_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_flow_magnitudes([])


def test_farneback_params_defaults_match_opencv_sample_usage() -> None:
    params = FarnebackParams()
    assert params.pyr_scale == 0.5
    assert params.levels == 3
    assert params.winsize == 15
