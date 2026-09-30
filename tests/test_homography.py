from __future__ import annotations

import numpy as np
import pytest

from module5_6.geometry import apply_homography
from module5_6.homography import (
    HomographyParams,
    estimate_homography,
    mean_reprojection_error,
    reprojection_errors,
    transform_points,
    validate_correspondences,
)

# A fixed set of four non-collinear planar points, reused across tests as software-verification
# fixtures - never presented as real assignment correspondences.
_SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def _many_points(rng: np.random.Generator, count: int = 30) -> np.ndarray:
    return rng.uniform(0.0, 100.0, size=(count, 2))


def test_validate_correspondences_accepts_four_non_collinear_points() -> None:
    validate_correspondences(_SQUARE, _SQUARE)


def test_validate_correspondences_rejects_fewer_than_four_points() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        validate_correspondences(_SQUARE[:3], _SQUARE[:3])


def test_validate_correspondences_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same shape"):
        validate_correspondences(_SQUARE, _SQUARE[:3])


def test_validate_correspondences_rejects_collinear_points() -> None:
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ValueError, match="collinear"):
        validate_correspondences(collinear, _SQUARE)


def test_estimate_homography_identity() -> None:
    result = estimate_homography(_SQUARE, _SQUARE, params=HomographyParams(method="all"))

    recovered = transform_points(result.matrix, _SQUARE)
    np.testing.assert_allclose(recovered, _SQUARE, atol=1e-6)
    assert np.all(result.inlier_mask)


def test_estimate_homography_pure_translation() -> None:
    translated = _SQUARE + np.array([5.0, -3.0])

    result = estimate_homography(_SQUARE, translated, params=HomographyParams(method="all"))

    recovered = transform_points(result.matrix, _SQUARE)
    np.testing.assert_allclose(recovered, translated, atol=1e-6)


def test_estimate_homography_rotation_and_scale() -> None:
    theta = np.pi / 6
    scale = 1.5
    rotation_scale = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed = _SQUARE @ rotation_scale.T + np.array([2.0, 1.0])

    result = estimate_homography(_SQUARE, transformed, params=HomographyParams(method="all"))

    recovered = transform_points(result.matrix, _SQUARE)
    np.testing.assert_allclose(recovered, transformed, atol=1e-6)


def test_estimate_homography_known_projective_transform() -> None:
    H_true = np.array([[1.2, 0.1, 3.0], [-0.05, 1.1, 2.0], [0.0007, 0.0004, 1.0]])
    rng = np.random.default_rng(0)
    points1 = _many_points(rng)
    points2 = apply_homography(H_true, points1)

    result = estimate_homography(points1, points2, params=HomographyParams(method="all"))

    recovered = transform_points(result.matrix, points1)
    np.testing.assert_allclose(recovered, points2, atol=1e-4)


def test_estimate_homography_exact_four_point_case() -> None:
    H_true = np.array([[1.0, 0.2, 4.0], [0.1, 1.0, -2.0], [0.0005, 0.0, 1.0]])
    points2 = apply_homography(H_true, _SQUARE)

    result = estimate_homography(_SQUARE, points2, params=HomographyParams(method="all"))

    recovered = transform_points(result.matrix, _SQUARE)
    np.testing.assert_allclose(recovered, points2, atol=1e-6)


def test_estimate_homography_with_noisy_correspondences_stays_close() -> None:
    H_true = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(1)
    points1 = _many_points(rng)
    points2 = apply_homography(H_true, points1) + rng.normal(scale=0.2, size=points1.shape)

    result = estimate_homography(points1, points2, params=HomographyParams(method="ransac", ransac_reproj_threshold=2.0))
    errors = reprojection_errors(result.matrix, points1, points2)

    assert mean_reprojection_error(errors) < 1.0


def test_estimate_homography_ransac_rejects_outliers() -> None:
    H_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(2)
    inlier_points1 = _many_points(rng, count=25)
    inlier_points2 = apply_homography(H_true, inlier_points1)
    outlier_points1 = rng.uniform(0, 100, size=(5, 2))
    outlier_points2 = rng.uniform(0, 100, size=(5, 2))  # unrelated to H_true

    points1 = np.vstack([inlier_points1, outlier_points1])
    points2 = np.vstack([inlier_points2, outlier_points2])

    result = estimate_homography(points1, points2, params=HomographyParams(method="ransac", ransac_reproj_threshold=1.0))

    assert np.all(result.inlier_mask[:25])
    assert not np.all(result.inlier_mask[25:])
    recovered = transform_points(result.matrix, inlier_points1)
    np.testing.assert_allclose(recovered, inlier_points2, atol=1e-3)


def test_estimate_homography_rejects_degenerate_configuration() -> None:
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    with pytest.raises(ValueError, match="collinear"):
        estimate_homography(collinear, _SQUARE)


def test_transform_points_matches_manual_homogeneous_math() -> None:
    H = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    points = np.array([[1.0, 1.0], [3.0, -1.0]])

    result = transform_points(H, points)

    np.testing.assert_allclose(result, [[2.0, 2.0], [6.0, -2.0]])


def test_reprojection_errors_match_manual_formula() -> None:
    H = np.eye(3)
    points1 = np.array([[0.0, 0.0], [10.0, 10.0]])
    points2 = np.array([[0.0, 0.0], [13.0, 14.0]])  # second point offset by (3, 4) -> error 5

    errors = reprojection_errors(H, points1, points2)

    np.testing.assert_allclose(errors, [0.0, 5.0])


def test_mean_reprojection_error_matches_manual_average() -> None:
    errors = np.array([1.0, 2.0, 3.0, 4.0])

    assert mean_reprojection_error(errors) == pytest.approx(2.5)


def test_mean_reprojection_error_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mean_reprojection_error(np.array([]))
