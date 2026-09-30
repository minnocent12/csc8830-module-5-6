"""Planar homography estimation and reprojection for Module 5-6 Structure From Motion.

Implements IMPLEMENTATION_PLAN.md Sections 18-19: given planar point correspondences
``(x_i, y_i) <-> (x_i', y_i')``, estimate the 3x3 projective transform ``H`` such that
``s [u, v, 1]^T ~ H [x, y, 1]^T``, then validate it by reprojecting each source point and
comparing against its observed correspondence. `cv2.findHomography` does the numerical
estimation; the homogeneous-coordinate application itself is `module5_6.geometry.apply_homography`,
not reimplemented here. This models a single flat/2D planar object's projective registration
between two views - not general (non-planar) multi-view 3D reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from module5_6.geometry import apply_homography, euclidean_distance

HomographyMethod = Literal["all", "ransac", "lmeds"]

_CV2_METHODS: dict[HomographyMethod, int] = {
    "all": 0,
    "ransac": cv2.RANSAC,
    "lmeds": cv2.LMEDS,
}


@dataclass(frozen=True)
class HomographyParams:
    """Parameters forwarded to ``cv2.findHomography``.

    ``method="all"`` uses every correspondence with no outlier rejection (plain least
    squares); ``"ransac"`` and ``"lmeds"`` are the two outlier-robust estimators OpenCV
    supports.
    """

    method: HomographyMethod = "ransac"
    ransac_reproj_threshold: float = 3.0
    max_iters: int = 2000
    confidence: float = 0.995


@dataclass(frozen=True)
class HomographyResult:
    """An estimated homography and, for robust methods, which correspondences were inliers."""

    matrix: np.ndarray  # (3, 3)
    inlier_mask: np.ndarray  # (N,) bool
    method: HomographyMethod


def _are_points_collinear(points: np.ndarray, *, tol: float = 1e-6) -> bool:
    """True if every point lies (within ``tol``) on a single line through the first two points."""
    if points.shape[0] < 3:
        return True
    origin = points[0]
    vectors = points[1:] - origin
    reference = vectors[np.argmax(np.linalg.norm(vectors, axis=1))]
    reference_norm = np.linalg.norm(reference)
    if reference_norm < tol:
        return True  # all points coincide with the first point
    cross = vectors[:, 0] * reference[1] - vectors[:, 1] * reference[0]
    return bool(np.all(np.abs(cross) <= tol * reference_norm))


def validate_correspondences(points1: np.ndarray, points2: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``points1``/``points2`` can support a homography estimate.

    Checks: equal length, at least four correspondences, and that ``points1`` are not all
    collinear (a homography is underdetermined for a degenerate/collinear point configuration).
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    if points1.shape != points2.shape:
        raise ValueError(f"points1 and points2 must have the same shape, got {points1.shape} and {points2.shape}")
    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError(f"points must be (N, 2) arrays, got shape {points1.shape}")
    if points1.shape[0] < 4:
        raise ValueError(f"at least 4 point correspondences are required, got {points1.shape[0]}")
    if _are_points_collinear(points1):
        raise ValueError("points1 are (nearly) collinear; a homography is underdetermined for a degenerate configuration")
    if _are_points_collinear(points2):
        raise ValueError("points2 are (nearly) collinear; a homography is underdetermined for a degenerate configuration")


def estimate_homography(points1: np.ndarray, points2: np.ndarray, *, params: HomographyParams | None = None) -> HomographyResult:
    """Estimate ``H`` such that ``points2 ~= apply_homography(H, points1)``.

    Raises ``ValueError`` from `validate_correspondences` for degenerate input, or if OpenCV
    itself cannot find a solution (e.g. a numerically degenerate configuration not caught by
    the collinearity check).
    """
    validate_correspondences(points1, points2)
    params = params or HomographyParams()
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    cv2_method = _CV2_METHODS[params.method]
    if cv2_method == 0:
        matrix, mask = cv2.findHomography(points1, points2, method=0)
    else:
        matrix, mask = cv2.findHomography(
            points1,
            points2,
            method=cv2_method,
            ransacReprojThreshold=params.ransac_reproj_threshold,
            maxIters=params.max_iters,
            confidence=params.confidence,
        )
    if matrix is None:
        raise ValueError("cv2.findHomography could not estimate a homography for this configuration")
    inlier_mask = np.ones(points1.shape[0], dtype=bool) if mask is None else mask.reshape(-1).astype(bool)
    return HomographyResult(matrix=matrix, inlier_mask=inlier_mask, method=params.method)


def transform_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to ``(N, 2)`` points; delegates to `module5_6.geometry.apply_homography`."""
    return apply_homography(H, points)


def reprojection_errors(H: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Per-point reprojection error ``|points2 - H . points1|_2`` (IMPLEMENTATION_PLAN.md Section 19)."""
    predicted = transform_points(H, points1)
    return euclidean_distance(predicted, np.asarray(points2, dtype=np.float64))


def mean_reprojection_error(errors: np.ndarray) -> float:
    """Mean of per-point reprojection errors."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        raise ValueError("errors must contain at least one value")
    return float(np.mean(errors))
