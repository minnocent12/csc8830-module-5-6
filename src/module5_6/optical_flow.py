"""Dense Farneback optical flow computation and visualization for Module 5-6.

These are pure array-in/array-out functions with no video file or Streamlit dependency, so
they are reused unchanged by ``module5_6.video``, standalone scripts, tests, and the web app.

Brightness constancy, the optical-flow constraint equation, and the Lucas-Kanade derivation
are documented separately (Phase 3, per IMPLEMENTATION_PLAN.md); this module only implements
the classical OpenCV Farneback dense-flow estimator itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from module5_6.types import FlowStatistics


@dataclass(frozen=True)
class FarnebackParams:
    """Parameters forwarded to ``cv2.calcOpticalFlowFarneback``.

    Defaults follow OpenCV's own Farneback sample usage. See the OpenCV documentation for the
    meaning of each field.
    """

    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


def compute_farneback_flow(
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    *,
    params: FarnebackParams | None = None,
) -> np.ndarray:
    """Compute a dense ``(H, W, 2)`` float32 optical-flow field between two grayscale frames.

    ``flow[y, x] = (u, v)`` is the estimated horizontal (``u``) and vertical (``v``) pixel
    displacement of the point at ``(x, y)`` in ``previous_gray`` as it moves into
    ``next_gray``, following the two-frame tracking setup
    ``I1(x, y) ~= I2(x + u, y + v)``.
    """
    if previous_gray.shape != next_gray.shape:
        raise ValueError("previous_gray and next_gray must have the same shape")
    if previous_gray.ndim != 2:
        raise ValueError("previous_gray and next_gray must be single-channel grayscale arrays")
    params = params or FarnebackParams()
    previous_uint8 = previous_gray if previous_gray.dtype == np.uint8 else previous_gray.astype(np.uint8)
    next_uint8 = next_gray if next_gray.dtype == np.uint8 else next_gray.astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        previous_uint8,
        next_uint8,
        None,
        params.pyr_scale,
        params.levels,
        params.winsize,
        params.iterations,
        params.poly_n,
        params.poly_sigma,
        params.flags,
    )
    return flow.astype(np.float32, copy=False)


def flow_magnitude_angle(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(magnitude, angle_degrees)`` from a dense ``(H, W, 2)`` flow field."""
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must be an (H, W, 2) array")
    magnitude, angle_degrees = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    return magnitude, angle_degrees


def flow_to_hsv_bgr(flow: np.ndarray, *, magnitude_scale: float | None = None) -> np.ndarray:
    """Visualize a dense flow field as an HSV-encoded BGR image.

    Hue encodes motion direction (OpenCV hue range 0-179 maps to 0-360 degrees). Value encodes
    motion magnitude, normalized by ``magnitude_scale`` if given, otherwise by this frame's own
    maximum magnitude. Saturation is fixed at maximum so direction and magnitude both stay
    visible even for small motions.
    """
    magnitude, angle_degrees = flow_magnitude_angle(flow)
    height, width = magnitude.shape
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(angle_degrees / 2.0, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    scale = magnitude_scale if magnitude_scale and magnitude_scale > 0 else float(magnitude.max())
    if scale > 0:
        normalized = np.clip((magnitude / scale) * 255.0, 0, 255)
    else:
        normalized = np.zeros_like(magnitude)
    hsv[..., 2] = normalized.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(
    frame_bgr: np.ndarray,
    flow: np.ndarray,
    *,
    step: int = 16,
    color: tuple[int, int, int] = (0, 255, 0),
    min_magnitude: float = 1.0,
) -> np.ndarray:
    """Overlay sampled flow-vector arrows on a copy of ``frame_bgr``; never mutates the input."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a 3-channel BGR array")
    if flow.shape[:2] != frame_bgr.shape[:2]:
        raise ValueError("flow and frame_bgr must share the same height and width")
    if step <= 0:
        raise ValueError("step must be positive")
    overlay = frame_bgr.copy()
    height, width = flow.shape[:2]
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            fx, fy = flow[y, x]
            if (fx * fx + fy * fy) ** 0.5 < min_magnitude:
                continue
            start = (x, y)
            end = (int(round(x + fx)), int(round(y + fy)))
            cv2.arrowedLine(overlay, start, end, color, 1, tipLength=0.4)
    return overlay


def summarize_flow_magnitudes(magnitudes: Sequence[np.ndarray]) -> FlowStatistics:
    """Aggregate per-pixel magnitude statistics across one or more flow fields."""
    if not magnitudes:
        raise ValueError("magnitudes must contain at least one array")
    stacked = np.concatenate([np.asarray(m, dtype=np.float64).reshape(-1) for m in magnitudes])
    return FlowStatistics(
        mean_magnitude=float(np.mean(stacked)),
        median_magnitude=float(np.median(stacked)),
        max_magnitude=float(np.max(stacked)),
        frame_pairs=len(magnitudes),
    )
