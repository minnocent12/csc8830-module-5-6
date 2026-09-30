from __future__ import annotations

import cv2
import numpy as np
import pytest

from module5_6.features import (
    FeatureMatch,
    MatchParams,
    ORBParams,
    detect_and_describe,
    match_descriptors,
    matched_coordinates,
)


def _textured_frame(size: tuple[int, int] = (240, 240), *, offset: int = 0) -> np.ndarray:
    """A deterministic textured grayscale frame; software verification only, not assignment data."""
    rng = np.random.default_rng(offset)
    noise = rng.integers(0, 256, size=size, dtype=np.uint8)
    # Add a few solid rectangles so ORB has real corner-like structure to find, not just noise.
    textured = noise.copy()
    textured[40:100, 40:100] = 220
    textured[150:210, 30:90] = 30
    textured[90:160, 150:220] = 180
    return textured


def _translated(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    height, width = frame.shape[:2]
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def test_detect_and_describe_finds_features_in_textured_image() -> None:
    frame = _textured_frame()

    points, descriptors = detect_and_describe(frame)

    assert points.shape[0] > 0
    assert points.shape[1] == 2
    assert descriptors.shape == (points.shape[0], 32)
    assert descriptors.dtype == np.uint8
    assert points.dtype == np.float32


def test_detect_and_describe_handles_blank_image_without_raising() -> None:
    blank = np.full((100, 100), 128, dtype=np.uint8)

    points, descriptors = detect_and_describe(blank)

    assert points.shape == (0, 2)
    assert descriptors.shape == (0, 32)


def test_detect_and_describe_rejects_color_input() -> None:
    with pytest.raises(ValueError, match="single-channel"):
        detect_and_describe(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detect_and_describe_respects_n_features_cap() -> None:
    frame = _textured_frame()

    points, _ = detect_and_describe(frame, params=ORBParams(n_features=5))

    assert points.shape[0] <= 5


def test_match_descriptors_recovers_known_translation() -> None:
    """A known synthetic (dx, dy) shift is software verification, not assignment evidence."""
    dx, dy = 6, -4
    previous = _textured_frame()
    following = _translated(previous, dx, dy)

    points1, descriptors1 = detect_and_describe(previous)
    points2, descriptors2 = detect_and_describe(following)
    assert points1.shape[0] > 5 and points2.shape[0] > 5

    matches = match_descriptors(descriptors1, descriptors2)
    assert len(matches) > 3

    query_xy, train_xy = matched_coordinates(points1, points2, matches)
    displacement = train_xy - query_xy
    # Interior matches (away from the border-replicated edges) should track the known shift.
    height, width = previous.shape
    margin = 20
    interior = (
        (query_xy[:, 0] >= margin)
        & (query_xy[:, 0] <= width - margin)
        & (query_xy[:, 1] >= margin)
        & (query_xy[:, 1] <= height - margin)
    )
    assert interior.sum() > 0
    assert float(np.median(displacement[interior, 0])) == pytest.approx(dx, abs=1.5)
    assert float(np.median(displacement[interior, 1])) == pytest.approx(dy, abs=1.5)


def test_match_descriptors_is_sorted_by_ascending_distance() -> None:
    previous = _textured_frame()
    following = _translated(previous, 3, 2)
    _, descriptors1 = detect_and_describe(previous)
    _, descriptors2 = detect_and_describe(following)

    matches = match_descriptors(descriptors1, descriptors2)

    distances = [m.distance for m in matches]
    assert distances == sorted(distances)


def test_match_descriptors_handles_empty_descriptor_sets() -> None:
    empty = np.zeros((0, 32), dtype=np.uint8)
    non_empty = np.zeros((5, 32), dtype=np.uint8)

    assert match_descriptors(empty, non_empty) == []
    assert match_descriptors(non_empty, empty) == []
    assert match_descriptors(empty, empty) == []


def test_match_descriptors_max_distance_and_max_matches_filters() -> None:
    previous = _textured_frame()
    following = _translated(previous, 4, 4)
    _, descriptors1 = detect_and_describe(previous)
    _, descriptors2 = detect_and_describe(following)

    unfiltered = match_descriptors(descriptors1, descriptors2)
    assert len(unfiltered) > 5

    capped = match_descriptors(descriptors1, descriptors2, params=MatchParams(max_matches=3))
    assert len(capped) == 3
    assert capped == unfiltered[:3]

    strict = match_descriptors(descriptors1, descriptors2, params=MatchParams(max_distance=0.0))
    assert all(m.distance <= 0.0 for m in strict)


def test_match_descriptors_ratio_test_recovers_known_translation() -> None:
    dx, dy = 5, -3
    previous = _textured_frame()
    following = _translated(previous, dx, dy)
    points1, descriptors1 = detect_and_describe(previous)
    points2, descriptors2 = detect_and_describe(following)
    assert points1.shape[0] > 5 and points2.shape[0] > 5

    matches = match_descriptors(descriptors1, descriptors2, params=MatchParams(ratio_test_threshold=0.75))
    assert len(matches) > 0

    query_xy, train_xy = matched_coordinates(points1, points2, matches)
    displacement = train_xy - query_xy
    height, width = previous.shape
    margin = 20
    interior = (
        (query_xy[:, 0] >= margin)
        & (query_xy[:, 0] <= width - margin)
        & (query_xy[:, 1] >= margin)
        & (query_xy[:, 1] <= height - margin)
    )
    assert interior.sum() > 0
    assert float(np.median(displacement[interior, 0])) == pytest.approx(dx, abs=1.5)
    assert float(np.median(displacement[interior, 1])) == pytest.approx(dy, abs=1.5)


def test_match_descriptors_ratio_test_is_stricter_than_unfiltered() -> None:
    previous = _textured_frame(offset=9)
    following = _translated(previous, 4, 4)
    _, descriptors1 = detect_and_describe(previous)
    _, descriptors2 = detect_and_describe(following)

    unfiltered = match_descriptors(descriptors1, descriptors2, params=MatchParams(cross_check=False))
    ratio_tested = match_descriptors(descriptors1, descriptors2, params=MatchParams(ratio_test_threshold=0.75))

    assert len(ratio_tested) <= len(unfiltered)


def test_matched_coordinates_extracts_expected_pairs() -> None:
    query_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    train_points = np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float32)
    matches = [FeatureMatch(query_index=2, train_index=0, distance=5.0), FeatureMatch(query_index=0, train_index=1, distance=1.0)]

    query_xy, train_xy = matched_coordinates(query_points, train_points, matches)

    np.testing.assert_array_equal(query_xy, [[2.0, 2.0], [0.0, 0.0]])
    np.testing.assert_array_equal(train_xy, [[10.0, 10.0], [11.0, 11.0]])


def test_matched_coordinates_handles_empty_matches() -> None:
    query_xy, train_xy = matched_coordinates(np.zeros((3, 2)), np.zeros((3, 2)), [])

    assert query_xy.shape == (0, 2)
    assert train_xy.shape == (0, 2)
