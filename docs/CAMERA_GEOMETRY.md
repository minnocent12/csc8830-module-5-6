# Camera Geometry: Pinhole Model, Homogeneous Coordinates, and the Planar Homography

This document derives the pinhole camera model and the planar-scene simplification
IMPLEMENTATION_PLAN.md Sections 15-16 describe, ending at the homography relation
`module5_6.homography` and `module5_6.sfm` implement. `docs/SFM_CALCULATIONS.md` continues
with the concrete estimation/reprojection workflow and a worked numerical example.

No numeric value in this document is measured from a real camera; it defines the general
symbols (`K`, `R`, `t`, `n`, `d`) that will be filled in with real, recorded values only once
the four real assignment images and any actually-available camera information exist
(`module5_6.camera.ViewMetadata`) - see `docs/STRUCTURE_FROM_MOTION_THEORY.md`.

## 1. Homogeneous coordinates

A 2D point `(x, y)` is represented in homogeneous coordinates as `[x, y, 1]^T` (or any nonzero
scalar multiple `[sx, sy, s]^T`, which represents the same point); a 3D point `(X, Y, Z)`
likewise as `[X, Y, Z, 1]^T`. Homogeneous coordinates let a projective transformation (which
includes translation, unlike a purely linear map) be written as a single matrix
multiplication, and let points "at infinity" (parallel lines meeting) be represented finitely
[1]. `module5_6.geometry.to_homogeneous`/`from_homogeneous` implement exactly this
conversion (append/drop the scale coordinate, dividing through on the way back).

## 2. The pinhole camera model

A 3D world point

```
P_w = [X, Y, Z, 1]^T
```

is projected to homogeneous image coordinates `[u, v, 1]^T` (up to scale `s`) by

```
s [u, v, 1]^T = K [R | t] P_w
```

where:

- **`K`** (3x3, the **intrinsic matrix**) encodes the camera's internal optics:

  ```
  K = [ fx   0   cx ]
      [  0  fy   cy ]
      [  0   0    1 ]
  ```

  `fx`, `fy` are the focal length in horizontal/vertical pixel units; `cx`, `cy` are the
  principal point (where the optical axis meets the image plane), typically near the image
  center but not necessarily exactly there.
- **`R`** (3x3 rotation matrix) and **`t`** (3x1 translation vector) together form the
  **extrinsic** parameters `[R | t]`: the camera's orientation and position relative to the
  world coordinate frame.
- **`s`** is the projective scale factor introduced by the homogeneous representation (it is
  divided out, per Section 1, to recover the actual pixel coordinate `(u, v)`).

This is the standard pinhole model; see Hartley and Zisserman [1], Chapter 6, or the OpenCV
camera-calibration documentation [2] for the equivalent formulation OpenCV's own calibration
and `findHomography`/`solvePnP` functions use. `module5_6.camera.intrinsic_matrix_from_values`
builds `K` from explicitly supplied `fx, fy, cx, cy` - never from image width/height alone,
since image size says nothing about the actual optics.

## 3. What can (and cannot) be known without calibration

Every extrinsic/intrinsic quantity above requires either a proper camera calibration procedure
or a real, recorded measurement/estimate at capture time:

- `K` requires calibration (e.g. a checkerboard calibration, as in Module 2) or a
  manufacturer-reported focal length converted to pixel units using the sensor's real pixel
  pitch - not simply invented.
- `R`, `t` (the camera's pose for a given view) are not directly observable without either a
  calibration rig, fiducial markers, or the homography-based recovery the planar
  simplification enables (Section 4 below, `H = K(R - t n^T / d) K^-1`, which still requires
  `K` to separate `R`, `t` from `H`).
- Incomplete EXIF metadata (if present in a real photo) may report a nominal focal length in
  millimeters, but that alone is not a verified intrinsic matrix - it says nothing about the
  sensor's pixel pitch, principal point, or lens distortion, and must not be treated as
  equivalent to calibration.

`module5_6.camera.ViewMetadata` reflects this directly: `focal_length_mm`, `intrinsic_matrix`,
`position_xyz`, `orientation`, and `distance_to_object_m` all default to `None` and are only
ever set to a caller-supplied value - never inferred from `width`/`height` or any other field.

## 4. Planar-object simplification: the induced homography

Because the assignment's object is flat, its surface can be modeled in its own coordinate
frame as the plane `Z = 0` (IMPLEMENTATION_PLAN.md Section 16). For two views of the same
plane, the mapping between their image coordinates is a single 3x3 homography `H`:

```
s x' = H x
```

where `x = [x, y, 1]^T`, `x' = [u, v, 1]^T` are homogeneous image coordinates in the two views.
For two pinhole cameras `K[I|0]` and `K[R|t]` viewing the plane with unit normal `n` and
(camera-to-plane) distance `d`, the induced homography is [1] (Chapter 13):

```
H = K ( R - t n^T / d ) K^-1
```

This is why a flat object lets Question 2 skip a fundamental/essential-matrix pipeline
entirely (IMPLEMENTATION_PLAN.md Section 21 is the *optional* non-planar alternative): the
homography `H` alone fully describes how the object's image warps from one view to the next,
and `H` can be estimated directly from point correspondences (`docs/SFM_CALCULATIONS.md`
Section 2) without ever separately solving for `K`, `R`, `t`, `n`, or `d`.

`module5_6.geometry.apply_homography` implements the point-mapping side of this
(`x' = H x`, normalized); `module5_6.homography.estimate_homography` implements the
correspondence-to-`H` estimation side.

## References

1. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.
   Cambridge University Press, 2004 (Chapter 6, camera models; Chapter 13, scene planes and
   homographies).
2. OpenCV, "Camera Calibration and 3D Reconstruction," OpenCV 4.x documentation,
   <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>.
