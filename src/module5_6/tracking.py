"""Shi-Tomasi feature detection and pyramidal Lucas-Kanade motion tracking for Module 5-6.

Pure array-in/array-out functions with no video file or Streamlit dependency, mirroring the
design of ``module5_6.optical_flow``, so the same core is reused by ``module5_6.video``
(not yet wired in this phase), tests, future scripts, and the web app.

Implements the two-frame tracking problem from IMPLEMENTATION_PLAN.md Section 10: for a point
P = (x, y) in a previous frame, find P' = (x + u, y + v) in the next frame such that
previous(x, y) ~= next(x + u, y + v). The formal brightness-constancy and Lucas-Kanade
least-squares derivations for the report are a later phase; this module only implements the
classical OpenCV estimator itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ShiTomasiParams:
    """Parameters forwarded to ``cv2.goodFeaturesToTrack``."""

    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: float = 7.0
    block_size: int = 7
    use_harris_detector: bool = False
    k: float = 0.04


@dataclass(frozen=True)
class LucasKanadeParams:
    """Parameters forwarded to ``cv2.calcOpticalFlowPyrLK``."""

    win_size: tuple[int, int] = (21, 21)
    max_level: int = 3
    max_iterations: int = 30
    epsilon: float = 0.01
    min_eig_threshold: float = 1e-4


def _lk_criteria(params: LucasKanadeParams) -> tuple[int, int, float]:
    return (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, params.max_iterations, params.epsilon)


def detect_features(gray: np.ndarray, *, params: ShiTomasiParams | None = None) -> np.ndarray:
    """Detect Shi-Tomasi corner features.

    Returns an ``(N, 1, 2)`` float32 array - OpenCV's native point shape for
    ``goodFeaturesToTrack``/``calcOpticalFlowPyrLK`` - or an ``(0, 1, 2)`` array if no corner
    meets ``params``.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a single-channel grayscale array")
    params = params or ShiTomasiParams()
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=params.max_corners,
        qualityLevel=params.quality_level,
        minDistance=params.min_distance,
        blockSize=params.block_size,
        useHarrisDetector=params.use_harris_detector,
        k=params.k,
    )
    if corners is None:
        return np.zeros((0, 1, 2), dtype=np.float32)
    return corners.astype(np.float32, copy=False)


def points_to_xy(points: np.ndarray) -> np.ndarray:
    """Reshape OpenCV's ``(N, 1, 2)`` point array to a plain ``(N, 2)`` array."""
    points = np.asarray(points)
    if points.ndim == 3 and points.shape[1] == 1 and points.shape[2] == 2:
        return points.reshape(-1, 2)
    if points.ndim == 2 and points.shape[1] == 2:
        return points
    raise ValueError(f"points must be shaped (N, 1, 2) or (N, 2), got {points.shape}")


def xy_to_points(xy: np.ndarray) -> np.ndarray:
    """Reshape a plain ``(N, 2)`` array to OpenCV's native ``(N, 1, 2)`` point array."""
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be shaped (N, 2), got {xy.shape}")
    return xy.reshape(-1, 1, 2)


@dataclass(frozen=True)
class TrackedPoints:
    """Pyramidal Lucas-Kanade tracking result for one previous -> next frame pair.

    ``status[i]`` is True where OpenCV reports ``previous_points[i]`` was successfully
    tracked to ``next_points[i]``; ``error[i]`` is OpenCV's per-point tracking error (only
    meaningful where ``status[i]`` is True).
    """

    previous_points: np.ndarray  # (N, 2) float32
    next_points: np.ndarray  # (N, 2) float32
    status: np.ndarray  # (N,) bool
    error: np.ndarray  # (N,) float32


def track_points(
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    points: np.ndarray,
    *,
    params: LucasKanadeParams | None = None,
) -> TrackedPoints:
    """Track ``points`` from ``previous_gray`` into ``next_gray`` with pyramidal Lucas-Kanade."""
    if previous_gray.shape != next_gray.shape:
        raise ValueError("previous_gray and next_gray must have the same shape")
    if previous_gray.ndim != 2:
        raise ValueError("previous_gray and next_gray must be single-channel grayscale arrays")
    previous_xy = points_to_xy(points)
    if previous_xy.shape[0] == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return TrackedPoints(
            previous_points=empty,
            next_points=empty.copy(),
            status=np.zeros((0,), dtype=bool),
            error=np.zeros((0,), dtype=np.float32),
        )
    params = params or LucasKanadeParams()
    previous_cv = xy_to_points(previous_xy)
    previous_uint8 = previous_gray if previous_gray.dtype == np.uint8 else previous_gray.astype(np.uint8)
    next_uint8 = next_gray if next_gray.dtype == np.uint8 else next_gray.astype(np.uint8)
    next_cv, status, error = cv2.calcOpticalFlowPyrLK(
        previous_uint8,
        next_uint8,
        previous_cv,
        None,
        winSize=params.win_size,
        maxLevel=params.max_level,
        criteria=_lk_criteria(params),
        minEigThreshold=params.min_eig_threshold,
    )
    return TrackedPoints(
        previous_points=previous_xy,
        next_points=points_to_xy(next_cv),
        status=status.reshape(-1).astype(bool),
        error=error.reshape(-1).astype(np.float32),
    )


def filter_valid_tracks(tracked: TrackedPoints) -> TrackedPoints:
    """Keep only the point pairs OpenCV's status output marks as successfully tracked."""
    mask = tracked.status
    return TrackedPoints(
        previous_points=tracked.previous_points[mask],
        next_points=tracked.next_points[mask],
        status=tracked.status[mask],
        error=tracked.error[mask],
    )


def displacement_vectors(tracked: TrackedPoints) -> np.ndarray:
    """Return each tracked point's ``(u, v)`` displacement: ``next_points - previous_points``."""
    return tracked.next_points - tracked.previous_points


def displacement_magnitude(displacements: np.ndarray) -> np.ndarray:
    """Return the Euclidean magnitude ``sqrt(u^2 + v^2)`` of each displacement vector."""
    displacements = np.asarray(displacements)
    if displacements.ndim != 2 or displacements.shape[1] != 2:
        raise ValueError("displacements must be an (N, 2) array")
    return np.linalg.norm(displacements, axis=1)


@dataclass(frozen=True)
class ForwardBackwardResult:
    """Forward-backward tracking-error validation (Kalal et al.'s forward-backward error).

    Tracks points forward (``previous -> next``) then backward (``next -> previous``); a
    small Euclidean distance between the original point and its backward-tracked position
    indicates a trustworthy track.
    """

    forward: TrackedPoints
    backward_points: np.ndarray  # (N, 2) float32, tracked from forward.next_points back
    backward_status: np.ndarray  # (N,) bool
    fb_error: np.ndarray  # (N,) float32, NaN where forward or backward tracking failed


def forward_backward_validate(
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    points: np.ndarray,
    *,
    params: LucasKanadeParams | None = None,
) -> ForwardBackwardResult:
    """Validate tracks by re-tracking backward and measuring drift from the original point."""
    forward = track_points(previous_gray, next_gray, points, params=params)
    backward = track_points(next_gray, previous_gray, forward.next_points, params=params)
    with np.errstate(invalid="ignore"):
        fb_error = np.linalg.norm(backward.next_points - forward.previous_points, axis=1)
    both_valid = forward.status & backward.status
    fb_error = np.where(both_valid, fb_error, np.nan)
    return ForwardBackwardResult(
        forward=forward,
        backward_points=backward.next_points,
        backward_status=backward.status,
        fb_error=fb_error.astype(np.float32),
    )


def valid_forward_backward_mask(
    result: ForwardBackwardResult, *, max_fb_error: float = 1.0
) -> np.ndarray:
    """Combine OpenCV status in both directions with a forward-backward error threshold."""
    with np.errstate(invalid="ignore"):
        error_ok = result.fb_error <= max_fb_error
    return result.forward.status & result.backward_status & error_ok


@dataclass
class Trajectory:
    """A single tracked feature's position history across consecutive frames.

    ``positions[0]`` is the detection-frame location. The trajectory stops growing once the
    point fails Lucas-Kanade tracking (``alive`` becomes False), but its partial history up to
    that point is kept for visualization.
    """

    point_id: int
    start_frame_index: int
    positions: list[tuple[float, float]] = field(default_factory=list)
    alive: bool = True


def track_trajectories(
    gray_frames: Sequence[np.ndarray],
    *,
    shi_tomasi_params: ShiTomasiParams | None = None,
    lk_params: LucasKanadeParams | None = None,
) -> list[Trajectory]:
    """Detect features on the first frame and track them across the whole frame sequence.

    Each initially detected feature becomes one ``Trajectory``. A point that fails Lucas-Kanade
    tracking on any consecutive frame pair is marked not alive and stops accumulating further
    positions; the trajectories of points still alive keep growing.
    """
    if len(gray_frames) < 2:
        raise ValueError("at least two frames are required to build a trajectory")
    initial_points = points_to_xy(detect_features(gray_frames[0], params=shi_tomasi_params))
    count = initial_points.shape[0]
    trajectories = [
        Trajectory(point_id=i, start_frame_index=0, positions=[tuple(initial_points[i])])
        for i in range(count)
    ]
    if count == 0:
        return trajectories

    current_points = initial_points.copy()
    alive = np.ones(count, dtype=bool)
    for frame_index in range(1, len(gray_frames)):
        if not alive.any():
            break
        alive_indices = np.where(alive)[0]
        tracked = track_points(
            gray_frames[frame_index - 1],
            gray_frames[frame_index],
            current_points[alive_indices],
            params=lk_params,
        )
        for local_index, global_index in enumerate(alive_indices):
            if tracked.status[local_index]:
                current_points[global_index] = tracked.next_points[local_index]
                trajectories[global_index].positions.append(tuple(current_points[global_index]))
            else:
                alive[global_index] = False
                trajectories[global_index].alive = False
    return trajectories


def draw_tracked_points(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 3,
) -> np.ndarray:
    """Draw a filled circle at each point on a copy of ``frame_bgr``; never mutates the input."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a 3-channel BGR array")
    overlay = frame_bgr.copy()
    for x, y in points_to_xy(points):
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius, color, -1)
    return overlay


def draw_displacement_vectors(
    frame_bgr: np.ndarray,
    previous_points: np.ndarray,
    next_points: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
    tip_length: float = 0.3,
) -> np.ndarray:
    """Draw an arrow from each previous point to its tracked next point."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a 3-channel BGR array")
    previous_xy = points_to_xy(previous_points)
    next_xy = points_to_xy(next_points)
    if previous_xy.shape[0] != next_xy.shape[0]:
        raise ValueError("previous_points and next_points must have the same length")
    overlay = frame_bgr.copy()
    for (x1, y1), (x2, y2) in zip(previous_xy, next_xy):
        start = (int(round(x1)), int(round(y1)))
        end = (int(round(x2)), int(round(y2)))
        cv2.arrowedLine(overlay, start, end, color, thickness, tipLength=tip_length)
    return overlay


def draw_trajectories(
    frame_bgr: np.ndarray,
    trajectories: Sequence[Trajectory],
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw each trajectory's position history as a polyline on a copy of ``frame_bgr``."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a 3-channel BGR array")
    overlay = frame_bgr.copy()
    for trajectory in trajectories:
        if len(trajectory.positions) < 2:
            continue
        polyline_points = np.array(trajectory.positions, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [polyline_points], isClosed=False, color=color, thickness=thickness)
    return overlay
