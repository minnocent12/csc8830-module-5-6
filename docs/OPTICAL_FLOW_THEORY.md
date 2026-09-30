# Optical Flow Theory: Brightness Constancy, the Optical-Flow Constraint, and the Aperture Problem

This document derives the optical-flow constraint equation from the brightness-constancy
assumption, then explains the aperture problem that motivates the Lucas-Kanade local
constant-motion assumption used in `docs/MOTION_TRACKING_DERIVATION.md`.

This is theoretical/report material. No numeric result in this document comes from a real
video; the two assignment videos and their required experimental validation remain
**PENDING USER EXPERIMENT** (see `docs/TRACKING_VALIDATION.md`, not yet created, for that
later phase).

## 1. Notation and coordinate convention

Let `I(x, y, t)` denote the image intensity at pixel column `x`, row `y`, and time `t`. This
matches the coordinate convention used throughout this codebase (`src/module5_6/video.py`,
`optical_flow.py`, `tracking.py`, `interpolation.py`): `x` increases rightward (column index),
`y` increases downward (row index), and `t` indexes video frames.

## 2. Brightness constancy assumption

The starting assumption of classical optical flow is that a point's brightness does not change
as it moves from one frame to the next - only its position changes. If a point at `(x, y)` at
time `t` moves by `(Δx, Δy)` during a small time interval `Δt`, brightness constancy states:

```
I(x, y, t) = I(x + Δx, y + Δy, t + Δt)
```

This is an idealization: it ignores illumination changes, specular highlights, and occlusion,
which is why optical flow becomes unreliable in those regions (see Section 6 of
`IMPLEMENTATION_PLAN.md`).

## 3. First-order Taylor expansion

Expand the right-hand side of the brightness-constancy equation to first order around
`(x, y, t)`, assuming `I` is differentiable and `(Δx, Δy, Δt)` is small:

```
I(x + Δx, y + Δy, t + Δt) ≈ I(x, y, t) + Ix·Δx + Iy·Δy + It·Δt
```

where `Ix = ∂I/∂x`, `Iy = ∂I/∂y`, and `It = ∂I/∂t` are the image's partial derivatives
(spatial gradients and the temporal derivative) evaluated at `(x, y, t)`.

## 4. The optical-flow constraint equation

Substitute the Taylor expansion into the brightness-constancy equation:

```
I(x, y, t) = I(x, y, t) + Ix·Δx + Iy·Δy + It·Δt
```

Subtract `I(x, y, t)` from both sides:

```
0 = Ix·Δx + Iy·Δy + It·Δt
```

Divide through by `Δt` and define the instantaneous velocity components:

```
u = Δx / Δt      v = Δy / Δt
```

`u` is the horizontal (x-direction) pixel velocity and `v` is the vertical (y-direction) pixel
velocity of the point between frames. Substituting gives the **optical-flow constraint
equation**, independently derived in this same form by Horn and Schunck [2] and by Lucas and
Kanade [1], who differ only in how they resolve the aperture problem below:

```
Ix·u + Iy·v + It = 0
```

This single scalar equation relates the measurable image gradients (`Ix`, `Iy`, `It`) to the
unknown motion `(u, v)` at that pixel. It is exactly the equation implemented for a full
`(H, W, 2)` flow field by `module5_6.optical_flow.compute_farneback_flow` (Phase 1), which
wraps OpenCV's `cv2.calcOpticalFlowFarneback` [3], and underlies the per-point tracking in
`module5_6.tracking.track_points` (Phase 2), even though neither function computes `Ix`, `Iy`,
`It` explicitly by name - see `docs/MOTION_TRACKING_DERIVATION.md` Section 6 for how the
library implementations relate to this pedagogical model.

## 5. Why one pixel is not enough: one equation, two unknowns

Rearranged, the constraint equation at a single pixel is:

```
Ix·u + Iy·v = -It
```

This is one linear equation in two unknowns, `u` and `v`. A single equation defines a *line*
in `(u, v)` space, not a unique point - only the component of motion **along the local
gradient direction** (`(Ix, Iy)`) is constrained; the component **perpendicular** to the
gradient is completely unconstrained by this one pixel. Additional information is required to
resolve `(u, v)` uniquely.

## 6. The aperture problem

This ambiguity is the classical **aperture problem**: viewed through a small aperture (here, a
single pixel's neighborhood), a moving edge's true motion cannot be distinguished from any
other motion with the same component along the edge's gradient direction. A textureless or
purely edge-like region (where the local gradient is one-dimensional) cannot, by itself, fix
both `u` and `v`.

Lucas and Kanade (1981) [1] resolve the aperture problem by assuming that **neighboring pixels
in a small window share approximately the same motion** `(u, v)`. That assumption turns one
equation per pixel into a whole system of equations over the window - more equations than
unknowns - which can be solved by least squares. The full derivation of that system, its
solution, and its relationship to the pyramidal OpenCV implementation used in Phase 2 is in
`docs/MOTION_TRACKING_DERIVATION.md`.

## References

1. B. D. Lucas and T. Kanade, "An Iterative Image Registration Technique with an Application
   to Stereo Vision," in *Proceedings of the 7th International Joint Conference on Artificial
   Intelligence (IJCAI'81)*, vol. 2, 1981, pp. 674-679.
2. B. K. P. Horn and B. G. Schunck, "Determining Optical Flow," *Artificial Intelligence*,
   vol. 17, no. 1-3, pp. 185-203, 1981.
3. OpenCV, "Optical Flow," OpenCV 4.x documentation,
   <https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html>.
