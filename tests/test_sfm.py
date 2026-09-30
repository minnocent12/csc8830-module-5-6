from __future__ import annotations

import cv2
import numpy as np
import pytest

from module5_6.features import MatchParams, ORBParams, detect_and_describe
from module5_6.geometry import apply_homography, transform_boundary_points
from module5_6.sfm import ViewRegistration, register_view, register_views


def _textured_reference(size: tuple[int, int] = (320, 320), *, seed: int = 0) -> np.ndarray:
    """A deterministic, richly textured grayscale image; software verification only."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=size, dtype=np.uint8)
    # Add several high-contrast rectangles so ORB has strong corner structure everywhere,
    # including after a moderate projective warp.
    frame[30:90, 30:90] = 230
    frame[220:280, 30:90] = 20
    frame[30:90, 220:280] = 20
    frame[220:280, 220:280] = 230
    frame[140:180, 140:180] = 128
    return frame


def _warp_to_view(reference: np.ndarray, H_true: np.ndarray) -> np.ndarray:
    """Build a synthetic 'view' image such that, for any point p in view coordinates, the same
    content sits in `reference` at `apply_homography(H_true, p)` - i.e. H_true maps this view's
    pixel coordinates into the reference frame, exactly the relationship `register_view` is
    meant to recover.
    """
    height, width = reference.shape[:2]
    H_inv = np.linalg.inv(H_true)
    return cv2.warpPerspective(reference, H_inv, (width, height), borderMode=cv2.BORDER_REPLICATE)


_H_TRANSLATION = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]])
_H_PROJECTIVE = np.array([[1.0, 0.02, 15.0], [-0.01, 1.0, -10.0], [0.0003, 0.0001, 1.0]])


def test_register_view_recovers_known_translation() -> None:
    """A known synthetic homography is software verification, not assignment evidence."""
    reference = _textured_reference()
    view = _warp_to_view(reference, _H_TRANSLATION)
    reference_keypoints, reference_descriptors = detect_and_describe(
        reference, params=ORBParams(n_features=300)
    )

    registration = register_view(
        reference_keypoints, reference_descriptors, "view_1", view, orb_params=ORBParams(n_features=300)
    )

    assert isinstance(registration, ViewRegistration)
    assert registration.mean_reprojection_error < 2.0

    height, width = reference.shape
    corners_view = np.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]])
    true_reference_corners = apply_homography(_H_TRANSLATION, corners_view)
    recovered_reference_corners = transform_boundary_points(registration.homography, corners_view)
    np.testing.assert_allclose(recovered_reference_corners, true_reference_corners, atol=3.0)


def test_register_view_recovers_known_projective_transform() -> None:
    reference = _textured_reference(seed=1)
    view = _warp_to_view(reference, _H_PROJECTIVE)
    reference_keypoints, reference_descriptors = detect_and_describe(
        reference, params=ORBParams(n_features=400)
    )

    registration = register_view(
        reference_keypoints, reference_descriptors, "view_1", view, orb_params=ORBParams(n_features=400)
    )

    assert registration.mean_reprojection_error < 3.0
    height, width = reference.shape
    corners_view = np.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]])
    true_reference_corners = apply_homography(_H_PROJECTIVE, corners_view)
    recovered_reference_corners = transform_boundary_points(registration.homography, corners_view)
    np.testing.assert_allclose(recovered_reference_corners, true_reference_corners, atol=5.0)


def test_register_view_transforms_supplied_boundary_points() -> None:
    reference = _textured_reference(seed=2)
    view = _warp_to_view(reference, _H_TRANSLATION)
    reference_keypoints, reference_descriptors = detect_and_describe(
        reference, params=ORBParams(n_features=300)
    )
    boundary_in_view = np.array([[50.0, 50.0], [270.0, 50.0], [270.0, 270.0], [50.0, 270.0]])

    registration = register_view(
        reference_keypoints,
        reference_descriptors,
        "view_1",
        view,
        orb_params=ORBParams(n_features=300),
        boundary_points_view=boundary_in_view,
    )

    assert registration.registered_boundary is not None
    expected = apply_homography(_H_TRANSLATION, boundary_in_view)
    np.testing.assert_allclose(registration.registered_boundary, expected, atol=3.0)


def test_register_view_boundary_is_none_when_not_supplied() -> None:
    reference = _textured_reference(seed=3)
    view = _warp_to_view(reference, _H_TRANSLATION)
    reference_keypoints, reference_descriptors = detect_and_describe(
        reference, params=ORBParams(n_features=300)
    )

    registration = register_view(
        reference_keypoints, reference_descriptors, "view_1", view, orb_params=ORBParams(n_features=300)
    )

    assert registration.registered_boundary is None


def test_register_view_applies_match_params_filter() -> None:
    reference = _textured_reference(seed=7)
    view = _warp_to_view(reference, _H_TRANSLATION)
    reference_keypoints, reference_descriptors = detect_and_describe(reference, params=ORBParams(n_features=300))

    unfiltered = register_view(
        reference_keypoints, reference_descriptors, "view_1", view, orb_params=ORBParams(n_features=300)
    )
    filtered = register_view(
        reference_keypoints,
        reference_descriptors,
        "view_1",
        view,
        orb_params=ORBParams(n_features=300),
        match_params=MatchParams(max_matches=5),
    )

    assert filtered.matched_points_view.shape[0] == 5
    assert filtered.matched_points_view.shape[0] < unfiltered.matched_points_view.shape[0]


def test_register_view_raises_with_insufficient_matches() -> None:
    reference = _textured_reference(seed=4)
    blank_view = np.full_like(reference, 128)
    reference_keypoints, reference_descriptors = detect_and_describe(reference)

    with pytest.raises(ValueError, match="at least 4"):
        register_view(reference_keypoints, reference_descriptors, "view_blank", blank_view)


def test_register_views_registers_multiple_views_and_builds_consensus_boundary() -> None:
    reference = _textured_reference(seed=5)
    view_translation = _warp_to_view(reference, _H_TRANSLATION)
    view_projective = _warp_to_view(reference, _H_PROJECTIVE)
    boundary_in_view = np.array([[60.0, 60.0], [260.0, 60.0], [260.0, 260.0], [60.0, 260.0]])
    reference_boundary = apply_homography(_H_TRANSLATION, boundary_in_view)  # arbitrary but known reference boundary

    result = register_views(
        "reference",
        reference,
        {"view_translation": view_translation, "view_projective": view_projective},
        orb_params=ORBParams(n_features=400),
        reference_boundary_points=reference_boundary,
        boundary_points_by_view={"view_translation": boundary_in_view, "view_projective": boundary_in_view},
    )

    assert result.reference_view_id == "reference"
    assert set(result.registrations.keys()) == {"view_translation", "view_projective"}
    assert result.consensus_boundary is not None
    assert result.consensus_boundary.shape == (4, 2)
    # The consensus is an unweighted mean of three boundary estimates (reference's own, plus
    # each registered view's), so it inherits some of the less-precise projective-view estimate;
    # this checks the mechanism lands in the right neighborhood, not sub-pixel agreement.
    np.testing.assert_allclose(result.consensus_boundary, reference_boundary, atol=10.0)


def test_register_views_with_no_boundary_points_has_no_consensus() -> None:
    reference = _textured_reference(seed=6)
    view = _warp_to_view(reference, _H_TRANSLATION)

    result = register_views("reference", reference, {"view_1": view}, orb_params=ORBParams(n_features=300))

    assert result.consensus_boundary is None
