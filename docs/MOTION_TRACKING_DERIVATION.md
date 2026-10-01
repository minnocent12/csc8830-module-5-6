# Motion Tracking Derivation: Lucas-Kanade, the Two-Frame Tracking Problem, and the OpenCV Implementation

This document continues from `docs/OPTICAL_FLOW_THEORY.md` (brightness constancy, the
optical-flow constraint equation, and the aperture problem) to derive the Lucas-Kanade
least-squares solution, state the two-frame tracking problem the assignment asks for, and
explain how that pedagogical derivation relates to the pyramidal `cv2.calcOpticalFlowPyrLK`
implementation used in Phase 2 (`src/module5_6/tracking.py`).

This is theoretical/report material; no tracked pixel coordinate or tracking error in this
document itself is from a real video. The two assignment videos and their required
two-consecutive-frame pixel-location validation are now **complete** - see
`docs/TRACKING_VALIDATION.md` for the real predicted-vs-observed results (4.628 px and
8.408 px) this derivation's two-frame tracking problem produced.

## 1. The two-frame tracking problem

Let `F1 = I(x, y, t)` be the first frame and `F2 = I(x, y, t + Δt)` be the next frame. A point
at `P = (x, y)` in `F1` is expected, under brightness constancy and small motion, to appear at

```
P' = (x + u, y + v)
```

in `F2`. The tracking problem is: **find `(u, v)` such that `F1(x, y) ≈ F2(x + u, y + v)`.**
This is exactly what `module5_6.tracking.track_points` computes for a set of points between
two consecutive frames (Phase 2), and it is the same `(u, v)` defined in
`docs/OPTICAL_FLOW_THEORY.md` Section 4.

## 2. From one equation to an overdetermined system

Section 5-6 of `docs/OPTICAL_FLOW_THEORY.md` showed that the optical-flow constraint equation
at a single pixel,

```
Ix·u + Iy·v = -It
```

is one equation in two unknowns (the aperture problem). Lucas and Kanade's resolution [1] is
the **local constant-motion assumption**: within a small window `W` around the point being
tracked, every pixel `p_i` is assumed to share the same `(u, v)`. Writing the constraint
equation at each of the `n` pixels `p_1, p_2, …, p_n` in the window:

```
Ix(p_1)·u + Iy(p_1)·v = -It(p_1)
Ix(p_2)·u + Iy(p_2)·v = -It(p_2)
                ⋮
Ix(p_n)·u + Iy(p_n)·v = -It(p_n)
```

This is `n` equations in the same 2 unknowns `(u, v)` - normally `n` is much larger than 2
(for example, a 21x21 window has 441 pixels), so the system is **overdetermined**.

## 3. Matrix form

Stack the equations as `A·v = b`, where `v = [u, v]ᵀ` is the unknown velocity (unfortunately
"v" names both the velocity vector and its second scalar component in the literature; this
document writes the vector as `v_vec` where the distinction matters), and

```
        ⎡ Ix(p_1)  Iy(p_1) ⎤              ⎡ -It(p_1) ⎤
        ⎢ Ix(p_2)  Iy(p_2) ⎥              ⎢ -It(p_2) ⎥
    A = ⎢    ⋮        ⋮    ⎥,         b = ⎢    ⋮     ⎥
        ⎣ Ix(p_n)  Iy(p_n) ⎦              ⎣ -It(p_n) ⎦
```

`A` is `n x 2` (one row per pixel in the window, one column per unknown) and `b` is `n x 1`.

## 4. Least-squares solution (normal equations)

Because `A·v_vec = b` is overdetermined (more rows than columns), it generally has no exact
solution; instead, Lucas-Kanade finds the `v_vec` that minimizes the sum of squared residuals
`‖A·v_vec - b‖²`. The standard least-squares solution multiplies both sides by `Aᵀ`:

```
Aᵀ·A·v_vec = Aᵀ·b
```

`AᵀA` is a `2 x 2` matrix, so when it is invertible:

```
v_vec = (AᵀA)⁻¹ · Aᵀ·b
```

Expanded, `AᵀA` is the **structure tensor** of the window:

```
        ⎡ Σ Ix²      Σ Ix·Iy ⎤
AᵀA  =  ⎢                     ⎥
        ⎣ Σ Ix·Iy    Σ Iy²    ⎦
```

with each sum taken over the `n` pixels in the window.

## 5. When the solution is unreliable: the eigenvalue/feature-quality connection

`AᵀA` is invertible exactly when its two eigenvalues `λ1 ≥ λ2 > 0` are both non-negligible.
Intuitively:

- If **both** eigenvalues are small, the window is nearly flat (low texture/gradient) - there
  is not enough information to determine motion in any direction.
- If **one** eigenvalue is small and the other large, the window looks like a straight edge -
  this is the aperture problem again, now visible directly in `AᵀA`: motion along the edge is
  poorly constrained even though motion across it is well constrained.
- If **both** eigenvalues are reasonably large, the window contains a corner-like structure in
  two independent gradient directions, and `(AᵀA)⁻¹` is well conditioned.

This is precisely the criterion Shi and Tomasi's "Good Features to Track" [2] use to select
points worth tracking in the first place: `cv2.goodFeaturesToTrack`
(`module5_6.tracking.detect_features`, Phase 2) ranks candidate corners by the smaller
eigenvalue of `AᵀA`, and OpenCV's `calcOpticalFlowPyrLK` exposes the same idea through its
`minEigThreshold` parameter (`module5_6.tracking.LucasKanadeParams.min_eig_threshold`), which
rejects a track outright when the window's smaller eigenvalue is too small to trust the
least-squares solution. Practically: feature detection and tracking reliability are two views
of the same structure-tensor conditioning problem.

## 6. Relationship to the OpenCV pyramidal implementation (Phase 2)

`module5_6.tracking.track_points` calls `cv2.calcOpticalFlowPyrLK`, which implements the
algorithm described by Bouguet [3], not the single-window least-squares solution above exactly
as written. The differences that matter for interpreting Phase 2's results:

- **Pyramidal, coarse-to-fine estimation.** The image is downsampled into a pyramid
  (`LucasKanadeParams.max_level` levels). Motion is estimated at the coarsest level first,
  then used to initialize and refine the estimate at each finer level. This lets the
  implementation track larger displacements than a single-resolution window could, while the
  per-level refinement is still the windowed least-squares problem derived above.
- **Iterative refinement.** At each pyramid level, the `Aᵀb` residual is recomputed and the
  estimate updated for several iterations (`LucasKanadeParams.max_iterations`) or until the
  update is smaller than `LucasKanadeParams.epsilon` - this is a Newton-Raphson-style
  refinement of the same linearized model, needed because the first-order Taylor expansion in
  `docs/OPTICAL_FLOW_THEORY.md` Section 3 is only locally accurate.
- **Per-point status and error.** `calcOpticalFlowPyrLK` reports a `status` flag and an `error`
  value per point (surfaced as `TrackedPoints.status` / `TrackedPoints.error` in
  `module5_6.tracking`); a point can be marked not-found when the window leaves the image or
  when the local structure tensor is too poorly conditioned (`minEigThreshold`), matching
  Section 5 above.

In short: the *model* implemented here is exactly the brightness-constancy /
local-constant-motion / least-squares derivation above; the *library* wraps that model in a
pyramidal, iterative numerical scheme for robustness and larger displacements. Neither
`compute_farneback_flow` (Farneback's polynomial-expansion method, Phase 1) nor
`calcOpticalFlowPyrLK` (Phase 2) computes `Ix`, `Iy`, `It`, `A`, or `b` by those exact names
internally - this document describes the mathematical model the library is solving, not its
internal source code.

## References

1. B. D. Lucas and T. Kanade, "An Iterative Image Registration Technique with an Application
   to Stereo Vision," in *Proceedings of the 7th International Joint Conference on Artificial
   Intelligence (IJCAI'81)*, vol. 2, 1981, pp. 674-679.
2. J. Shi and C. Tomasi, "Good Features to Track," in *Proceedings of the IEEE Conference on
   Computer Vision and Pattern Recognition (CVPR)*, 1994, pp. 593-600.
3. J.-Y. Bouguet, "Pyramidal Implementation of the Lucas Kanade Feature Tracker: Description
   of the Algorithm," Intel Corporation, Microprocessor Research Labs, 2000.
4. OpenCV, "cv2.calcOpticalFlowPyrLK" and "Shi-Tomasi Corner Detector & Good Features to
   Track," OpenCV 4.x documentation,
   <https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html> and
   <https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html>.

See also `docs/OPTICAL_FLOW_THEORY.md` for the brightness-constancy assumption, the
optical-flow constraint equation, and the aperture problem this derivation builds on.
