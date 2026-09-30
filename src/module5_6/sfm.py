"""Planar Structure-From-Motion / homography-registration coordination layer.

Coordinates `module5_6.features` (correspondences), `module5_6.homography` (registration),
and `module5_6.geometry` (point/boundary transformation) into the workflow
IMPLEMENTATION_PLAN.md Sections 16-20 describe for a flat/2D planar object: pick a reference
view, estimate a homography from each other view into the reference frame, identify inliers,
transform matched points (and, if supplied, the object's boundary corners) into the reference
frame, and compute reprojection error. No UI logic lives here.

Terminology note: this computes a 2D planar homography registration for a flat object, per the
assignment's explicit planar-object simplification (IMPLEMENTATION_PLAN.md Section 16) - not a
dense/full 3D reconstruction. Nothing here recovers 3D structure, camera pose in 3D, or depth;
see IMPLEMENTATION_PLAN.md Section 21 for the optional (not implemented in this phase)
epipolar/triangulation enhancement that would.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module5_6.features import ORBParams, detect_and_describe, match_descriptors, matched_coordinates
from module5_6.homography import HomographyParams, estimate_homography, reprojection_errors
from module5_6.geometry import transform_boundary_points


@dataclass(frozen=True)
class ViewRegistration:
    """One non-reference view's homography registration into the reference view's frame."""

    view_id: str
    homography: np.ndarray  # (3, 3): maps this view's pixel coordinates into the reference frame
    inlier_mask: np.ndarray  # (N,) bool, over matched_points_view/matched_points_reference
    matched_points_view: np.ndarray  # (N, 2) matched keypoints in this view
    matched_points_reference: np.ndarray  # (N, 2) corresponding matched keypoints in the reference view
    reprojection_errors: np.ndarray  # (N,) per matched point, over ALL matches (inliers and outliers)
    mean_reprojection_error: float  # mean over inlier matches only (see register_view docstring)
    registered_boundary: np.ndarray | None  # this view's boundary points, mapped into the reference frame


def register_view(
    reference_keypoints: np.ndarray,
    reference_descriptors: np.ndarray,
    view_id: str,
    view_gray: np.ndarray,
    *,
    orb_params: ORBParams | None = None,
    homography_params: HomographyParams | None = None,
    boundary_points_view: np.ndarray | None = None,
) -> ViewRegistration:
    """Detect features on one view, match them to an already-detected reference, and register it.

    ``ViewRegistration.mean_reprojection_error`` averages over inlier matches only (per
    ``HomographyParams.method``'s inlier mask) - with the default ``"ransac"`` method, a
    descriptor mismatch the estimator correctly flagged as an outlier should not be allowed to
    dominate the reported error. ``ViewRegistration.reprojection_errors`` itself still reports
    every matched point, inliers and outliers alike, for diagnostic/visualization use.

    Raises ``ValueError`` (from `module5_6.homography.validate_correspondences`) if fewer than
    four matches are found, or if the matched points are degenerate/collinear.
    """
    view_keypoints, view_descriptors = detect_and_describe(view_gray, params=orb_params)
    matches = match_descriptors(view_descriptors, reference_descriptors)
    if len(matches) < 4:
        raise ValueError(
            f"only {len(matches)} descriptor match(es) found for view {view_id!r}; "
            "at least 4 are required to estimate a homography"
        )
    points_view, points_reference = matched_coordinates(view_keypoints, reference_keypoints, matches)
    result = estimate_homography(points_view, points_reference, params=homography_params)
    errors = reprojection_errors(result.matrix, points_view, points_reference)
    inlier_errors = errors[result.inlier_mask]
    # Mean over inliers only: a robust method (ransac/lmeds) deliberately excludes outlier
    # matches from the fitted model, so averaging over every match (including the ones the
    # method itself flagged as not fitting) would defeat the point of outlier rejection and
    # let a handful of bad matches dominate the reported error. Falls back to all matches only
    # if, degenerately, no point was flagged inlier.
    mean_error = float(np.mean(inlier_errors)) if inlier_errors.size > 0 else float(np.mean(errors))
    registered_boundary = (
        transform_boundary_points(result.matrix, boundary_points_view)
        if boundary_points_view is not None
        else None
    )
    return ViewRegistration(
        view_id=view_id,
        homography=result.matrix,
        inlier_mask=result.inlier_mask,
        matched_points_view=points_view,
        matched_points_reference=points_reference,
        reprojection_errors=errors,
        mean_reprojection_error=mean_error,
        registered_boundary=registered_boundary,
    )


@dataclass(frozen=True)
class PlanarSfMResult:
    """The full planar registration result: one reference view plus every other registered view."""

    reference_view_id: str
    registrations: dict[str, ViewRegistration]
    consensus_boundary: np.ndarray | None  # simple mean of all available registered boundaries


def _consensus_boundary(
    registrations: dict[str, ViewRegistration], *, reference_boundary: np.ndarray | None
) -> np.ndarray | None:
    """Average every available registered boundary (reference's own, plus each registered view's).

    This is a simple, transparent combination - an unweighted mean of whichever boundaries are
    available - not a bundle adjustment or any other optimization; documented as such so it is
    not mistaken for a more sophisticated multi-view estimator.
    """
    candidates = [reference_boundary] if reference_boundary is not None else []
    candidates += [r.registered_boundary for r in registrations.values() if r.registered_boundary is not None]
    if not candidates:
        return None
    return np.mean(np.stack(candidates, axis=0), axis=0)


def register_views(
    reference_view_id: str,
    reference_gray: np.ndarray,
    other_views_gray: dict[str, np.ndarray],
    *,
    orb_params: ORBParams | None = None,
    homography_params: HomographyParams | None = None,
    reference_boundary_points: np.ndarray | None = None,
    boundary_points_by_view: dict[str, np.ndarray] | None = None,
) -> PlanarSfMResult:
    """Register every view in ``other_views_gray`` into ``reference_gray``'s coordinate frame.

    ``boundary_points_by_view`` (keyed by view id) supplies each non-reference view's own
    boundary-corner points, if the caller has them; ``reference_boundary_points`` supplies the
    reference view's boundary directly (already in the reference frame, so it needs no
    transform). A view that fails to register (raises inside `register_view`) is re-raised -
    callers that want partial results should catch per-view rather than calling this directly
    for an experiment where some views may not register.
    """
    reference_keypoints, reference_descriptors = detect_and_describe(reference_gray, params=orb_params)
    boundary_points_by_view = boundary_points_by_view or {}
    registrations: dict[str, ViewRegistration] = {}
    for view_id, view_gray in other_views_gray.items():
        registrations[view_id] = register_view(
            reference_keypoints,
            reference_descriptors,
            view_id,
            view_gray,
            orb_params=orb_params,
            homography_params=homography_params,
            boundary_points_view=boundary_points_by_view.get(view_id),
        )
    consensus_boundary = _consensus_boundary(registrations, reference_boundary=reference_boundary_points)
    return PlanarSfMResult(
        reference_view_id=reference_view_id,
        registrations=registrations,
        consensus_boundary=consensus_boundary,
    )
