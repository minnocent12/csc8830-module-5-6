"""ORB feature detection and descriptor matching for Module 5-6 planar Structure From Motion.

Pure array-in/array-out functions with no file or Streamlit dependency, mirroring the design
of `module5_6.optical_flow` and `module5_6.tracking`. Provides the "automatic correspondences"
path of IMPLEMENTATION_PLAN.md Section 17 for planar homography estimation
(`module5_6.homography`, `module5_6.sfm`). The "manual validation points" path in that same
section is supplied directly by the caller/UI and does not go through this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

_DESCRIPTOR_BYTES = 32  # cv2.ORB's descriptor width for the default WTA_K=2


@dataclass(frozen=True)
class ORBParams:
    """Parameters forwarded to ``cv2.ORB_create``."""

    n_features: int = 500
    scale_factor: float = 1.2
    n_levels: int = 8
    edge_threshold: int = 31
    fast_threshold: int = 20


def detect_and_describe(gray: np.ndarray, *, params: ORBParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Detect ORB keypoints and compute their descriptors.

    Returns ``(points, descriptors)``: ``points`` is an ``(N, 2)`` float32 array of keypoint
    ``(x, y)`` pixel coordinates; ``descriptors`` is an ``(N, 32)`` uint8 array. A blank or
    low-texture image (no detectable corners) returns ``(0, 2)``/``(0, 32)`` empty arrays
    rather than raising.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a single-channel grayscale array")
    params = params or ORBParams()
    orb = cv2.ORB_create(
        nfeatures=params.n_features,
        scaleFactor=params.scale_factor,
        nlevels=params.n_levels,
        edgeThreshold=params.edge_threshold,
        fastThreshold=params.fast_threshold,
    )
    gray_uint8 = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
    keypoints, descriptors = orb.detectAndCompute(gray_uint8, None)
    if not keypoints or descriptors is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, _DESCRIPTOR_BYTES), dtype=np.uint8)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points, descriptors.astype(np.uint8, copy=False)


@dataclass(frozen=True)
class FeatureMatch:
    """One descriptor correspondence between a query set and a train set."""

    query_index: int
    train_index: int
    distance: float


@dataclass(frozen=True)
class MatchParams:
    """Parameters controlling descriptor matching.

    ``ratio_test_threshold``, if set, switches matching to Lowe's ratio test (each query
    descriptor's two nearest train neighbors are found via ``knnMatch``, and the match is kept
    only if the best distance is below ``ratio_test_threshold`` times the second-best distance
    - typically ``0.75``). This targets a different failure mode than ``cross_check``/
    ``max_distance``: a query descriptor with several similarly-close train candidates (e.g.
    repeated/self-similar text glyphs or graphic texture) produces an ambiguous "best" match
    that a plain distance cutoff cannot detect, since the best distance alone can still look
    good. ``cross_check`` is ignored when ``ratio_test_threshold`` is set, because OpenCV's
    ``BFMatcher.knnMatch`` does not support ``crossCheck``.
    """

    cross_check: bool = True
    max_distance: float | None = None
    max_matches: int | None = None
    ratio_test_threshold: float | None = None


def match_descriptors(
    query_descriptors: np.ndarray, train_descriptors: np.ndarray, *, params: MatchParams | None = None
) -> list[FeatureMatch]:
    """Match ORB descriptors with a brute-force Hamming-distance matcher.

    Returns matches sorted by ascending distance (deterministic given deterministic input
    descriptors). Returns an empty list, without raising, when either descriptor set is empty.
    ``params.max_distance`` (if given) drops matches above that Hamming distance;
    ``params.max_matches`` (if given) keeps only the closest N matches after that filter.
    ``params.ratio_test_threshold`` (if given) instead applies Lowe's ratio test - see
    ``MatchParams`` - and is applied before ``max_distance``/``max_matches``.
    """
    params = params or MatchParams()
    if query_descriptors.shape[0] == 0 or train_descriptors.shape[0] == 0:
        return []
    if params.ratio_test_threshold is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(query_descriptors, train_descriptors, k=2)
        matches = [
            FeatureMatch(query_index=pair[0].queryIdx, train_index=pair[0].trainIdx, distance=float(pair[0].distance))
            for pair in knn
            if len(pair) == 2 and pair[0].distance < params.ratio_test_threshold * pair[1].distance
        ]
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=params.cross_check)
        cv_matches = matcher.match(query_descriptors, train_descriptors)
        matches = [
            FeatureMatch(query_index=m.queryIdx, train_index=m.trainIdx, distance=float(m.distance))
            for m in cv_matches
        ]
    matches.sort(key=lambda m: m.distance)
    if params.max_distance is not None:
        matches = [m for m in matches if m.distance <= params.max_distance]
    if params.max_matches is not None:
        matches = matches[: params.max_matches]
    return matches


def matched_coordinates(
    query_points: np.ndarray, train_points: np.ndarray, matches: Sequence[FeatureMatch]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the corresponding ``(N, 2)`` coordinate arrays for a list of matches."""
    if not matches:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    query_xy = np.array([query_points[m.query_index] for m in matches], dtype=np.float32)
    train_xy = np.array([train_points[m.train_index] for m in matches], dtype=np.float32)
    return query_xy, train_xy
