# Planar SfM Calculations: Homography Estimation, Reprojection, and Boundary Registration

Continues from `docs/CAMERA_GEOMETRY.md` (the pinhole model and the planar-scene homography
relation) with the concrete estimation workflow IMPLEMENTATION_PLAN.md Sections 17-20
describe, and a deterministic worked numerical example. Implementation:
`module5_6.features` (correspondences), `module5_6.homography` (estimation/reprojection),
`module5_6.geometry` (point transforms), `module5_6.sfm` (multi-view coordination).

The worked example in Section 4 uses invented point coordinates chosen to be easy to verify -
it is a mathematical example, not a measurement from any real assignment image. Real four-view
calculations remain **PENDING USER EXPERIMENT** until the actual images exist (see
`docs/STRUCTURE_FROM_MOTION_THEORY.md`).

## 1. Point correspondence

Two ways to obtain corresponding points between two views of the planar object
(IMPLEMENTATION_PLAN.md Section 17), both supported:

- **Automatic**: ORB keypoint detection plus brute-force Hamming-distance descriptor matching
  (`module5_6.features.detect_and_describe`, `match_descriptors`) [1][2].
- **Manual**: the user identifies and records the pixel coordinates of the same recognizable
  corner/feature across views directly - useful for the required hand-worked mathematics,
  since the exact coordinates are then known values rather than an algorithm's output.

## 2. Homography estimation (Direct Linear Transform)

Given a correspondence `(x, y) <-> (u, v)`, the projective relation
`[u, v, 1]^T ~ H [x, y, 1]^T` means the two sides are parallel (equal up to scale), so their
cross product is zero: `x' x (H x) = 0`. Expanding this cross-product constraint for one point
gives two independent linear equations in the 9 unknown entries of `H`
(`h11, h12, ..., h33`):

```
-x h11 - y h12 - h13 + 0 + 0 + 0 + (u x) h31 + (u y) h32 + u h33 = 0
0 + 0 + 0 - x h21 - y h22 - h23 + (v x) h31 + (v y) h32 + v h33 = 0
```

Stacking these two equations for each of `n` correspondences gives a `2n x 9` matrix `A` such
that `A h = 0`, where `h = [h11, ..., h33]^T`. Because `H` is only defined up to scale, the
solution is the right singular vector of `A` corresponding to its smallest singular value (the
**Direct Linear Transform**, DLT) [3]. At least 4 correspondences are required (8 equations for
8 degrees of freedom, fixing overall scale); the correspondences must not be collinear, or `A`
becomes rank-deficient and the system is underdetermined
(`module5_6.homography.validate_correspondences`).

`cv2.findHomography` [4] implements this DLT solution directly (`method="all"` in
`module5_6.homography.HomographyParams`), or a RANSAC/LMedS-wrapped robust version
(`method="ransac"`/`"lmeds"`, the default) that repeatedly fits a DLT solution to a random
minimal subset of correspondences and keeps the fit with the most inliers - discarding
mismatched correspondences (e.g. from imperfect ORB matches) rather than letting them corrupt
the estimate. `module5_6.homography.estimate_homography` calls this; the mathematical relation
above is what it is computing, not merely "calling a library function."

## 3. Reprojection validation

After estimating `H`, the predicted location of each correspondence's source point is
`p_hat' = H p_i` (normalized from homogeneous coordinates,
`module5_6.geometry.apply_homography`). The reprojection error is the Euclidean distance to the
actually observed corresponding point:

```
e_i = || p_i' - p_hat_i' ||_2
```

(`module5_6.homography.reprojection_errors`), and the mean reprojection error
(`module5_6.homography.mean_reprojection_error`) summarizes fit quality across all inlier
correspondences (IMPLEMENTATION_PLAN.md Section 19; `module5_6.sfm.ViewRegistration` computes
this mean over RANSAC/LMedS inliers only, so a correspondence the robust estimator itself
flagged as an outlier does not dominate the reported error - see that module's docstring).

## 4. Boundary registration across views

For a rectangular planar object with known boundary/corner points `P1, P2, P3, P4` in a given
view, the same homography that registers matched features also registers the boundary:
`transform_boundary_points(H, [P1, P2, P3, P4])` (IMPLEMENTATION_PLAN.md Section 20). Once
every other view's boundary has been mapped into the reference view's frame,
`module5_6.sfm.register_views` combines them into one consensus boundary estimate by a simple,
transparent **unweighted mean** across all available registered-boundary estimates (the
reference's own boundary, if supplied directly, plus every other view's registered boundary) -
not a bundle adjustment or any other optimization, and documented as such so it is not mistaken
for one.

## 5. Complete numerical worked example

Four correspondences between a "view" image and a reference image, and a deliberately simple
known homography `H_true`, chosen only to make the DLT setup easy to verify by hand:

```
H_true = [ 1  0   5 ]
         [ 0  1  -3 ]
         [ 0  0   1 ]
```

(a pure translation by `(5, -3)` - the simplest nontrivial homography, with `h31 = h32 = 0`).
Four non-collinear source points (a 10x10 square) and their true images under `H_true`:

| Point | `(x, y)` | `(u, v) = H_true (x, y)` |
| ----- | -------- | ------------------------- |
| P1    | (0, 0)    | (5, -3)                   |
| P2    | (10, 0)   | (15, -3)                  |
| P3    | (10, 10)  | (15, 7)                   |
| P4    | (0, 10)   | (5, 7)                    |

**Setting up the DLT system for P1** (`x=0, y=0, u=5, v=-3`), substituting into Section 2's
per-point equations:

```
-0*h11 - 0*h12 - h13 + 0 + 0 + 0 + (5*0)h31 + (5*0)h32 + 5*h33 = 0  ->  -h13 + 5 h33 = 0
0 + 0 + 0 - 0*h21 - 0*h22 - h23 + (-3*0)h31 + (-3*0)h32 + (-3)*h33 = 0  ->  -h23 - 3 h33 = 0
```

Fixing the overall scale with `h33 = 1` (valid since `H` is only defined up to scale) gives
`h13 = 5`, `h23 = -3` directly from P1 alone - consistent with `H_true`'s translation column.
Repeating for P2, P3, P4 and solving the full `8 x 9` system (by Gaussian elimination or, in
practice, the SVD-based solver `cv2.findHomography` uses) recovers the complete matrix:

```
h11 = 1, h12 = 0, h13 = 5
h21 = 0, h22 = 1, h23 = -3
h31 = 0, h32 = 0, h33 = 1
```

which is exactly `H_true`, confirming the setup. **Reprojection check** (Section 3), using
`H_true` on each source point:

```
e_i = || H_true p_i - p_i' ||_2 = 0  for i = 1, 2, 3, 4
```

(zero error, since these four correspondences were generated exactly from `H_true` with no
noise - a real experiment's correspondences will not fit perfectly, and the resulting nonzero
`e_i` values and their mean are exactly what Section 3 requires reporting).

This exact worked example - the same four points, the same `(5, -3)` translation, and zero
reprojection error - is checked against `module5_6.homography.estimate_homography` in
`tests/test_homography.py::test_estimate_homography_pure_translation`. The same DLT method
(not this specific hand-workable numeric case, since its points are randomly generated for
broader coverage) is additionally verified against a true projective `H_true` with nonzero
`h31, h32` in `tests/test_homography.py::test_estimate_homography_known_projective_transform`,
and end-to-end from synthetic images (detection through matching through registration) in
`tests/test_sfm.py::test_register_view_recovers_known_projective_transform`. All of this is
software verification against invented, exact numbers - not a real assignment measurement.

## References

1. E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, "ORB: An Efficient Alternative to SIFT
   or SURF," in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*,
   2011, pp. 2564-2571.
2. OpenCV, "ORB (Oriented FAST and Rotated BRIEF)," OpenCV 4.x documentation,
   <https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html>.
3. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.
   Cambridge University Press, 2004 (Section 4.1, the Direct Linear Transformation algorithm).
4. OpenCV, "Camera Calibration and 3D Reconstruction" (`cv2.findHomography`), OpenCV 4.x
   documentation, <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>.

See also `docs/CAMERA_GEOMETRY.md` for the pinhole camera model and the planar-scene
homography relation this estimation recovers, and `docs/STRUCTURE_FROM_MOTION_THEORY.md` for
the overall Question 2 workflow and terminology.
