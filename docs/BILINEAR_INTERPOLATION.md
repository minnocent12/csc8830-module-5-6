# Bilinear Interpolation: Derivation and Worked Example

This document derives bilinear interpolation from two sequential 1D linear interpolations and
works a complete numerical example. The implementation it describes is
`module5_6.interpolation.bilinear_interpolate` (Phase 3), demonstrated interactively on the
**Bilinear Interpolation & Theory** Streamlit page.

The worked example in Section 4 is a deterministic mathematical example, computed from chosen
numbers to illustrate the method - it is not derived from, and does not represent, any measured
pixel location from a real assignment video.

## 1. Why interpolation is needed

A tracked point's estimated location (for example, the `(u, v)` displacement produced by
Lucas-Kanade tracking in `docs/MOTION_TRACKING_DERIVATION.md`) is a floating-point coordinate,
not generally aligned to the integer pixel grid an image is stored on. To read an intensity
value "at" that estimated location - or to compare a predicted subpixel location against
something computed from the image - a rule is needed for combining the intensities of the
pixels that surround it. Bilinear interpolation is the standard first-order rule for this: a
smooth, differentiable estimate built from the four nearest integer pixels [1].

## 2. Setup: four neighboring pixels and fractional offsets

Suppose the estimated coordinate is `(x, y)`, lying between four integer-pixel neighbors:

```
(x0, y0)    (x1, y0)
(x0, y1)    (x1, y1)
```

with `x1 = x0 + 1` and `y1 = y0 + 1` (using this codebase's coordinate convention: `x`
increases rightward, `y` increases downward - see `module5_6.interpolation`). Define the
fractional offsets:

```
α = x - x0        (how far x is past x0, toward x1; α ∈ [0, 1))
β = y - y0        (how far y is past y0, toward y1; β ∈ [0, 1))
```

Let `I00 = I(x0, y0)`, `I10 = I(x1, y0)`, `I01 = I(x0, y1)`, `I11 = I(x1, y1)` be the four
known pixel intensities.

## 3. Derivation from two sequential 1D linear interpolations

**Step 1 - interpolate along x, at each of the two known rows.** A 1D linear interpolation
between two values `A` and `B` at fractional offset `α` is `(1-α)·A + α·B`. Apply this along
the top row (`y = y0`) and the bottom row (`y = y1`) separately:

```
R0 = (1 - α)·I00 + α·I10        (interpolated value along the row at y = y0)
R1 = (1 - α)·I01 + α·I11        (interpolated value along the row at y = y1)
```

`R0` and `R1` are now two values, both located at the same horizontal position `x`, one at row
`y0` and one at row `y1`.

**Step 2 - interpolate along y, between the two row results.** Apply a second 1D linear
interpolation, this time between `R0` and `R1` at fractional offset `β`:

```
I(x, y) = (1 - β)·R0 + β·R1
```

**Step 3 - substitute and expand.** Substituting `R0` and `R1`:

```
I(x, y) = (1 - β)·[(1 - α)·I00 + α·I10] + β·[(1 - α)·I01 + α·I11]
```

Distributing the multiplication gives the final bilinear interpolation formula:

```
I(x, y) = (1-α)(1-β)·I00 + α(1-β)·I10 + (1-α)β·I01 + αβ·I11
```

This matches IMPLEMENTATION_PLAN.md Section 11 exactly, and is implemented directly (not just
called from a library) by `module5_6.interpolation.bilinear_interpolate` and
`bilinear_interpolate_corners`. The four coefficients
`w00=(1-α)(1-β)`, `w10=α(1-β)`, `w01=(1-α)β`, `w11=αβ` are the **bilinear weights**: each is
the area of the rectangle on the *opposite* side of `(x, y)` from its corner, scaled so the
four weights sum to exactly 1 (`bilinear_weights` in `module5_6.interpolation`).

## 4. Complete numerical worked example

Let the four known pixel intensities be `I00 = 10`, `I10 = 20`, `I01 = 30`, `I11 = 40` (chosen
round numbers for a clear worked example, as also used in `IMPLEMENTATION_PLAN.md` Section 26's
testing strategy), and let the fractional offsets be `α = 0.3`, `β = 0.7`.

**Step 1 - interpolate along x:**

```
R0 = (1 - 0.3)·10 + 0.3·20 = 0.7·10 + 0.3·20 = 7.0 + 6.0 = 13.0
R1 = (1 - 0.3)·30 + 0.3·40 = 0.7·30 + 0.3·40 = 21.0 + 12.0 = 33.0
```

**Step 2 - interpolate along y:**

```
I(x, y) = (1 - 0.7)·13.0 + 0.7·33.0 = 0.3·13.0 + 0.7·33.0 = 3.9 + 23.1 = 27.0
```

**Cross-check with the expanded weight formula:**

```
w00 = (1-0.3)(1-0.7) = 0.7 * 0.3 = 0.21
w10 = 0.3 * (1-0.7)  = 0.3 * 0.3 = 0.09
w01 = (1-0.3) * 0.7  = 0.7 * 0.7 = 0.49
w11 = 0.3 * 0.7               = 0.21

I(x, y) = 0.21*10 + 0.09*20 + 0.49*30 + 0.21*40
        = 2.1 + 1.8 + 14.7 + 8.4
        = 27.0
```

Both routes agree: **`I(x, y) = 27.0`**. This exact example is checked by
`tests/test_interpolation.py::test_bilinear_interpolate_corners_matches_manual_calculation`
against `module5_6.interpolation.bilinear_interpolate_corners(10, 20, 30, 40, 0.3, 0.7)`, and
is reproduced interactively on the Bilinear Interpolation & Theory page.

## 5. Boundary behavior

`bilinear_interpolate` requires `(x, y)` to lie within `[0, width-1] x [0, height-1]` - the
only region where all four neighboring pixels are guaranteed to exist inside the image. A
coordinate exactly on the last row or column (e.g. `x = width - 1`) is valid: its "upper"
neighbor is clamped to the same last index, `α` (or `β`) is then exactly 0, and the formula
correctly reduces to the single known pixel value - verified by
`tests/test_interpolation.py::test_bilinear_weights_reduce_to_single_corner_at_grid_points` and
`test_bilinear_interpolate_at_integer_coordinate_returns_exact_pixel`. Coordinates outside
`[0, width-1] x [0, height-1]` raise `ValueError` rather than silently extrapolating or
clamping, since no well-defined four-neighbor lookup exists there.

## References

1. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. Pearson, image
   interpolation (bilinear interpolation).
2. OpenCV, `cv2.remap` / `INTER_LINEAR` documentation (used only for cross-validation in
   `tests/test_interpolation.py`, never as the educational implementation itself),
   <https://docs.opencv.org/4.x/>.

See also `docs/MOTION_TRACKING_DERIVATION.md` for why a tracked point's location is generally
subpixel/fractional in the first place.
