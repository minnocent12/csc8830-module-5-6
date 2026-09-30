from __future__ import annotations

import numpy as np
import pytest

from module5_6.geometry import (
    apply_homography,
    boundary_polygon_closed,
    euclidean_distance,
    from_homogeneous,
    register_view_to_reference,
    to_homogeneous,
    transform_boundary_points,
)


def test_to_homogeneous_appends_ones() -> None:
    points = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = to_homogeneous(points)

    np.testing.assert_array_equal(result, [[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]])


def test_from_homogeneous_normalizes_by_last_coordinate() -> None:
    points_h = np.array([[2.0, 4.0, 2.0], [9.0, 3.0, 3.0]])

    result = from_homogeneous(points_h)

    np.testing.assert_allclose(result, [[1.0, 2.0], [3.0, 1.0]])


def test_to_and_from_homogeneous_roundtrip() -> None:
    points = np.array([[5.0, -2.0], [0.0, 10.0], [3.5, 3.5]])

    roundtrip = from_homogeneous(to_homogeneous(points))

    np.testing.assert_allclose(roundtrip, points)


def test_from_homogeneous_rejects_point_at_infinity() -> None:
    with pytest.raises(ValueError, match="infinity"):
        from_homogeneous(np.array([[1.0, 1.0, 0.0]]))


def test_euclidean_distance_matches_manual_calculation() -> None:
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[3.0, 4.0], [1.0, 1.0]])

    distances = euclidean_distance(a, b)

    np.testing.assert_allclose(distances, [5.0, 0.0])


def test_euclidean_distance_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        euclidean_distance(np.zeros((2, 2)), np.zeros((3, 2)))


def test_apply_homography_identity_is_unchanged() -> None:
    points = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = apply_homography(np.eye(3), points)

    np.testing.assert_allclose(result, points)


def test_apply_homography_pure_translation() -> None:
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.0, 0.0], [10.0, 10.0]])

    result = apply_homography(H, points)

    np.testing.assert_allclose(result, [[5.0, -3.0], [15.0, 7.0]])


def test_apply_homography_matches_manual_projective_transform() -> None:
    """A hand-computed projective example: software verification, not assignment data."""
    H = np.array([[2.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.001, 0.0, 1.0]])
    point = np.array([[10.0, 5.0]])

    result = apply_homography(H, point)

    denom = 0.001 * 10.0 + 1.0
    expected_x = (2.0 * 10.0 + 1.0) / denom
    expected_y = 5.0 / denom
    np.testing.assert_allclose(result, [[expected_x, expected_y]])


def test_apply_homography_rejects_non_3x3_matrix() -> None:
    with pytest.raises(ValueError, match="3x3"):
        apply_homography(np.eye(2), np.array([[1.0, 2.0]]))


def test_transform_boundary_points_matches_apply_homography() -> None:
    H = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    boundary = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    result = transform_boundary_points(H, boundary)

    np.testing.assert_allclose(result, apply_homography(H, boundary))


def test_register_view_to_reference_matches_apply_homography() -> None:
    H = np.array([[1.0, 0.0, -2.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    points = np.array([[1.0, 1.0], [2.0, 2.0]])

    result = register_view_to_reference(H, points)

    np.testing.assert_allclose(result, apply_homography(H, points))


def test_boundary_polygon_closed_repeats_first_point() -> None:
    boundary = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    closed = boundary_polygon_closed(boundary)

    assert closed.shape == (5, 2)
    np.testing.assert_array_equal(closed[0], closed[-1])
    np.testing.assert_array_equal(closed[0], boundary[0])


def test_boundary_polygon_closed_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        boundary_polygon_closed(np.zeros((0, 2)))
