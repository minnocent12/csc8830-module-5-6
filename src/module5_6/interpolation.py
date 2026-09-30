"""Bilinear interpolation for Module 5-6.

An inspectable, from-scratch implementation of bilinear interpolation, derived from two
sequential 1D linear interpolations - see ``docs/BILINEAR_INTERPOLATION.md`` for the full
derivation and a deterministic worked numerical example. This module exists because the
assignment asks to derive/describe bilinear interpolation, not just call a library function;
OpenCV's own interpolation (``cv2.remap`` with ``INTER_LINEAR``) is used only for
cross-validation in tests, never to replace this implementation.

Coordinate convention (matches the rest of this codebase: video frames, optical flow, and
tracking): ``(x, y)`` are continuous image coordinates where ``x`` increases rightward (the
column index) and ``y`` increases downward (the row index) - i.e. ``image[y, x]`` for the
NumPy array backing an image. Given a fractional coordinate ``(x, y)``:

    x0 = floor(x), y0 = floor(y), x1 = x0 + 1, y1 = y0 + 1
    alpha = x - x0  (fractional offset in x, in [0, 1))
    beta  = y - y0  (fractional offset in y, in [0, 1))

with the four contributing neighbor pixels ``I00 = I(x0, y0)``, ``I10 = I(x1, y0)``,
``I01 = I(x0, y1)``, ``I11 = I(x1, y1)``, matching IMPLEMENTATION_PLAN.md Section 11 and
docs/BILINEAR_INTERPOLATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bilinear_weights(alpha: float, beta: float) -> tuple[float, float, float, float]:
    """Return the four bilinear weights ``(w00, w10, w01, w11)`` for offsets in ``[0, 1]``.

    Implements ``I(x,y) = (1-a)(1-b)*I00 + a(1-b)*I10 + (1-a)*b*I01 + a*b*I11``: the weights
    themselves, independent of any particular image or pixel values.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    weight_00 = (1.0 - alpha) * (1.0 - beta)
    weight_10 = alpha * (1.0 - beta)
    weight_01 = (1.0 - alpha) * beta
    weight_11 = alpha * beta
    return weight_00, weight_10, weight_01, weight_11


def bilinear_interpolate_corners(
    i00: float, i10: float, i01: float, i11: float, alpha: float, beta: float
) -> float:
    """Interpolate directly from four known corner values and fractional offsets.

    Useful for the report's numerical worked example, where the four pixel intensities are
    given directly rather than read from an image array.
    """
    weight_00, weight_10, weight_01, weight_11 = bilinear_weights(alpha, beta)
    return weight_00 * i00 + weight_10 * i10 + weight_01 * i01 + weight_11 * i11


@dataclass(frozen=True)
class BilinearSample:
    """The four contributing pixels, their weights, and the interpolated result at ``(x, y)``.

    Field names follow IMPLEMENTATION_PLAN.md Section 11 and docs/BILINEAR_INTERPOLATION.md.
    """

    x: float
    y: float
    x0: int
    y0: int
    x1: int
    y1: int
    alpha: float
    beta: float
    i00: float
    i10: float
    i01: float
    i11: float
    weight_00: float
    weight_10: float
    weight_01: float
    weight_11: float
    value: float


def bilinear_interpolate(image: np.ndarray, x: float, y: float) -> BilinearSample:
    """Interpolate a scalar/grayscale image at fractional coordinate ``(x, y)``.

    Implements ``I(x,y) = (1-a)(1-b)*I00 + a(1-b)*I10 + (1-a)*b*I01 + a*b*I11`` directly (see
    the module docstring and docs/BILINEAR_INTERPOLATION.md for the coordinate convention and
    derivation from two sequential 1D linear interpolations).

    Raises ``ValueError`` if ``(x, y)`` falls outside ``[0, width-1] x [0, height-1]``, since a
    well-defined four-neighbor lookup requires a valid integer pixel on every side.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 2:
        raise ValueError("image must be a single-channel (scalar/grayscale) 2D array")
    height, width = image.shape
    if width < 2 or height < 2:
        raise ValueError("image must be at least 2x2 for bilinear interpolation")
    if not (0.0 <= x <= width - 1) or not (0.0 <= y <= height - 1):
        raise ValueError(
            f"({x}, {y}) is outside the valid interpolation range "
            f"[0, {width - 1}] x [0, {height - 1}]"
        )

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    # Clamp the "upper" neighbor so it stays a valid index when x or y lands exactly on the
    # last integer coordinate; alpha/beta are then exactly 0, so I10/I01/I11 do not affect
    # the interpolated value, but they must still name a real pixel.
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    alpha = float(x - x0)
    beta = float(y - y0)

    i00 = float(image[y0, x0])
    i10 = float(image[y0, x1])
    i01 = float(image[y1, x0])
    i11 = float(image[y1, x1])

    weight_00, weight_10, weight_01, weight_11 = bilinear_weights(alpha, beta)
    value = weight_00 * i00 + weight_10 * i10 + weight_01 * i01 + weight_11 * i11

    return BilinearSample(
        x=float(x),
        y=float(y),
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        alpha=alpha,
        beta=beta,
        i00=i00,
        i10=i10,
        i01=i01,
        i11=i11,
        weight_00=weight_00,
        weight_10=weight_10,
        weight_01=weight_01,
        weight_11=weight_11,
        value=value,
    )
