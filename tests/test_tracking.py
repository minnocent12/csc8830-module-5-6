from __future__ import annotations

import cv2
import numpy as np
import pytest

from module5_6.tracking import (
    ForwardBackwardResult,
    LucasKanadeParams,
    ShiTomasiParams,
    TrackedPoints,
    detect_features,
    displacement_magnitude,
    displacement_vectors,
    draw_displacement_vectors,
    draw_tracked_points,
    draw_trajectories,
    filter_valid_tracks,
    forward_backward_validate,
    points_to_xy,
    track_points,
    track_trajectories,
    valid_forward_backward_mask,
    xy_to_points,
)


def _textured_frame(size: tuple[int, int] = (160, 160), *, offset: int = 0) -> np.ndarray:
    """A deterministic textured grayscale frame; software verification only, not assignment data."""
    rng = np.random.default_rng(offset)
    return rng.integers(0, 256, size=size, dtype=np.uint8)


def _translated(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    height, width = frame.shape[:2]
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def _interior_points(points: np.ndarray, width: int, height: int, margin: int = 25) -> np.ndarray:
    """Keep only points whose Lucas-Kanade window stays fully inside the frame after motion."""
    xy = points_to_xy(points)
    mask = (
        (xy[:, 0] >= margin)
        & (xy[:, 0] <= width - margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 1] <= height - margin)
    )
    return xy_to_points(xy[mask])


def test_detect_features_finds_corners_in_textured_image() -> None:
    frame = _textured_frame()

    points = detect_features(frame, params=ShiTomasiParams(max_corners=50, quality_level=0.1))

    assert points.shape[1:] == (1, 2)
    assert points.shape[0] > 5
    assert points.dtype == np.float32


def test_detect_features_returns_empty_for_flat_image() -> None:
    flat = np.full((64, 64), 128, dtype=np.uint8)

    points = detect_features(flat)

    assert points.shape == (0, 1, 2)


def test_detect_features_rejects_color_input() -> None:
    with pytest.raises(ValueError, match="single-channel"):
        detect_features(np.zeros((10, 10, 3), dtype=np.uint8))


def test_points_xy_roundtrip() -> None:
    xy = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    points = xy_to_points(xy)
    roundtrip = points_to_xy(points)

    assert points.shape == (2, 1, 2)
    np.testing.assert_array_equal(roundtrip, xy)


def test_points_to_xy_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shaped"):
        points_to_xy(np.zeros((3, 3)))


def test_track_points_recovers_known_translation() -> None:
    """A known synthetic (dx, dy) = (4, -3) shift is software verification, not assignment data."""
    dx, dy = 4, -3
    previous = _textured_frame()
    following = _translated(previous, dx, dy)
    height, width = previous.shape
    raw_points = detect_features(previous, params=ShiTomasiParams(max_corners=80, quality_level=0.1))
    points = _interior_points(raw_points, width, height)
    assert points.shape[0] > 5

    tracked = track_points(previous, following, points)
    valid = filter_valid_tracks(tracked)

    assert valid.previous_points.shape[0] > 0
    displacements = displacement_vectors(valid)
    assert float(np.median(displacements[:, 0])) == pytest.approx(dx, abs=1.0)
    assert float(np.median(displacements[:, 1])) == pytest.approx(dy, abs=1.0)


def test_track_points_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        track_points(
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 11), dtype=np.uint8),
            xy_to_points(np.array([[1.0, 1.0]])),
        )


def test_track_points_handles_empty_point_set() -> None:
    previous = _textured_frame()
    following = _textured_frame(offset=1)

    tracked = track_points(previous, following, np.zeros((0, 1, 2), dtype=np.float32))

    assert tracked.previous_points.shape == (0, 2)
    assert tracked.next_points.shape == (0, 2)
    assert tracked.status.shape == (0,)


def test_track_points_marks_out_of_bounds_points_invalid() -> None:
    """A point whose Lucas-Kanade window cannot fit near the frame edge is reported invalid."""
    previous = _textured_frame(size=(64, 64))
    following = _translated(previous, 5, 5)
    corner_point = xy_to_points(np.array([[0.0, 0.0]], dtype=np.float32))

    tracked = track_points(previous, following, corner_point, params=LucasKanadeParams(win_size=(21, 21)))

    assert tracked.status[0] == False  # noqa: E712 - explicit bool check reads clearer here


def test_filter_valid_tracks_keeps_only_successful_points() -> None:
    tracked = TrackedPoints(
        previous_points=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        next_points=np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]], dtype=np.float32),
        status=np.array([True, False, True]),
        error=np.array([0.1, 0.9, 0.2], dtype=np.float32),
    )

    valid = filter_valid_tracks(tracked)

    assert valid.previous_points.shape[0] == 2
    np.testing.assert_array_equal(valid.previous_points, [[0.0, 0.0], [2.0, 2.0]])


def test_displacement_vectors_and_magnitude_match_manual_calculation() -> None:
    tracked = TrackedPoints(
        previous_points=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        next_points=np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32),
        status=np.array([True, True]),
        error=np.array([0.0, 0.0], dtype=np.float32),
    )

    displacements = displacement_vectors(tracked)
    magnitudes = displacement_magnitude(displacements)

    np.testing.assert_array_equal(displacements, [[3.0, 4.0], [0.0, 1.0]])
    np.testing.assert_allclose(magnitudes, [5.0, 1.0])


def test_displacement_magnitude_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="N, 2"):
        displacement_magnitude(np.zeros((3, 3)))


def test_forward_backward_validate_reports_low_error_for_consistent_translation() -> None:
    dx, dy = 3, 2
    previous = _textured_frame()
    following = _translated(previous, dx, dy)
    height, width = previous.shape
    points = _interior_points(
        detect_features(previous, params=ShiTomasiParams(max_corners=80, quality_level=0.1)),
        width,
        height,
    )
    assert points.shape[0] > 5

    result = forward_backward_validate(previous, following, points)
    mask = valid_forward_backward_mask(result, max_fb_error=1.0)

    assert isinstance(result, ForwardBackwardResult)
    assert mask.sum() > 0
    assert float(np.nanmean(result.fb_error[mask])) < 1.0


def test_forward_backward_validate_rejects_some_tracks_between_unrelated_frames() -> None:
    previous = _textured_frame(offset=1)
    unrelated = _textured_frame(offset=99)
    points = detect_features(previous, params=ShiTomasiParams(max_corners=80, quality_level=0.1))
    assert points.shape[0] > 5

    result = forward_backward_validate(previous, unrelated, points)
    mask = valid_forward_backward_mask(result, max_fb_error=1.0)

    assert mask.sum() < points.shape[0]


def test_track_trajectories_builds_history_across_frames() -> None:
    dx, dy = 2, 1
    frames = [_textured_frame(size=(200, 200))]
    for step in range(1, 6):
        frames.append(_translated(frames[0], dx * step, dy * step))

    trajectories = track_trajectories(
        frames, shi_tomasi_params=ShiTomasiParams(max_corners=60, quality_level=0.1)
    )

    assert len(trajectories) > 0
    assert any(len(trajectory.positions) == len(frames) for trajectory in trajectories)
    for trajectory in trajectories:
        assert 1 <= len(trajectory.positions) <= len(frames)
        assert trajectory.positions[0] is not None


def test_track_trajectories_rejects_too_few_frames() -> None:
    with pytest.raises(ValueError, match="at least two frames"):
        track_trajectories([_textured_frame()])


def test_track_trajectories_handles_no_detected_features() -> None:
    flat_frames = [np.full((32, 32), 100, dtype=np.uint8) for _ in range(3)]

    trajectories = track_trajectories(flat_frames)

    assert trajectories == []


def test_draw_tracked_points_does_not_mutate_input() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    original = frame.copy()
    points = xy_to_points(np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32))

    overlay = draw_tracked_points(frame, points)

    np.testing.assert_array_equal(frame, original)
    assert not np.array_equal(overlay, frame)


def test_draw_displacement_vectors_rejects_length_mismatch() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    previous_points = xy_to_points(np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32))
    next_points = xy_to_points(np.array([[1.0, 1.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="same length"):
        draw_displacement_vectors(frame, previous_points, next_points)


def test_draw_trajectories_skips_single_point_trajectories() -> None:
    from module5_6.tracking import Trajectory

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    original = frame.copy()
    short_trajectory = Trajectory(point_id=0, start_frame_index=0, positions=[(5.0, 5.0)])

    overlay = draw_trajectories(frame, [short_trajectory])

    np.testing.assert_array_equal(overlay, original)
