"""Planar 2D geometry helpers for Module 5-6 Structure From Motion.

Homogeneous-coordinate conversion, Euclidean distance, and point/boundary transformation
through a homography, used to register a planar object's boundary across the four assignment
viewpoints (IMPLEMENTATION_PLAN.md Section 20). These are pure, cv2-independent numeric
primitives; `module5_6.homography` builds on `apply_homography` here rather than
reimplementing the homogeneous-coordinate math.
"""
from __future__ import annotations

import numpy as np


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Append a column of ones: ``(N, 2) -> (N, 3)``."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be an (N, 2) array, got shape {points.shape}")
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.hstack([points, ones])


def from_homogeneous(points_h: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Normalize by the last coordinate and drop it: ``(N, 3) -> (N, 2)``.

    Raises ``ValueError`` for any point whose homogeneous scale is within ``eps`` of zero -
    such a point is at infinity under this transform and has no finite 2D image.
    """
    points_h = np.asarray(points_h, dtype=np.float64)
    if points_h.ndim != 2 or points_h.shape[1] != 3:
        raise ValueError(f"points_h must be an (N, 3) array, got shape {points_h.shape}")
    scale = points_h[:, 2]
    if np.any(np.abs(scale) < eps):
        raise ValueError("one or more points are at infinity (homogeneous scale ~= 0)")
    return points_h[:, :2] / scale[:, None]


def euclidean_distance(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Per-point Euclidean distance between two ``(N, 2)`` point sets."""
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    if points_a.shape != points_b.shape:
        raise ValueError(f"points_a and points_b must have the same shape, got {points_a.shape} and {points_b.shape}")
    if points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError(f"points must be (N, 2) arrays, got shape {points_a.shape}")
    return np.linalg.norm(points_a - points_b, axis=1)


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` points through a ``3x3`` homography: ``x' = H x`` (homogeneous, normalized).

    This is the one place the projective transform `s*x' = H*x` (IMPLEMENTATION_PLAN.md
    Section 16/18) is actually computed; `module5_6.homography.transform_points` and every
    boundary/registration helper below delegate here.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H must be a 3x3 matrix, got shape {H.shape}")
    points_h = to_homogeneous(points)
    transformed_h = (H @ points_h.T).T
    return from_homogeneous(transformed_h)


def transform_boundary_points(H: np.ndarray, boundary_points: np.ndarray) -> np.ndarray:
    """Map an ordered set of object-boundary/corner points through a homography.

    Semantically the same operation as `apply_homography`, named for the boundary-recovery
    use case in IMPLEMENTATION_PLAN.md Section 20 (recovering `P1, P2, P3, P4` in a reference
    view's coordinate system).
    """
    return apply_homography(H, boundary_points)


def register_view_to_reference(H_view_to_reference: np.ndarray, points_in_view: np.ndarray) -> np.ndarray:
    """Map points from a secondary view's pixel coordinates into the reference view's frame.

    Semantically the same operation as `apply_homography`, named for the multi-view
    registration step in IMPLEMENTATION_PLAN.md Section 20 ("Correspondences from the four
    camera views can be transformed into a common reference coordinate system").
    """
    return apply_homography(H_view_to_reference, points_in_view)


def boundary_polygon_closed(points: np.ndarray) -> np.ndarray:
    """Return boundary points with the first point repeated at the end, for drawing a closed polygon."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be an (N, 2) array, got shape {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points must contain at least one point")
    return np.vstack([points, points[0:1]])
