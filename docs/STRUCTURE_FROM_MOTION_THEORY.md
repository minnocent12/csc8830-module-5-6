# Structure From Motion Theory (Question 2)

This document gives the conceptual overview of Question 2's four-view planar Structure From
Motion task and states precisely what this implementation does and does not claim. The
underlying camera/homography mathematics is in `docs/CAMERA_GEOMETRY.md`; the concrete
calculation workflow and a worked numerical example are in `docs/SFM_CALCULATIONS.md`.

**Status: COMPLETE (Phase 6).** The four real images (`data/sfm/view_1/IMG_7283.JPG` through
`view_4/IMG_7286.JPG`) have been captured and processed end-to-end by
`scripts/process_sfm_experiment.py`: real ORB correspondences, real RANSAC homographies, real
reprojection error, and real boundary registration. Every number in this section comes from
that run (`results/sfm/sfm_summary.json`); nothing here is invented. (Phase 5, which this
document originally described, built and tested the reusable software foundation only, against
synthetic fixtures - that software is what Phase 6 then ran on the real images.)

## 0. Phase 6 real-experiment summary

- **Object**: the front cover of a paperback book (*The Tempest*, William Shakespeare, Folger
  Shakespeare Library "Updated Edition" - a yellow/black textured, printed cover), treated as
  the flat/2D planar object per the assignment's planar simplification. The spine and page
  edges visible in the oblique views are not part of the tracked plane.
- **Reference view**: `view_1` (`IMG_7283.JPG`) - the centered/front viewpoint closest to the
  object (~9 in, per the user-recorded capture notes), chosen for having the least perspective
  foreshortening of the front cover face; see `docs/SFM_CALCULATIONS.md` Section 6 for the full
  per-view registration numbers.
- **Camera**: all four photos were taken with the same device and lens (Apple iPhone 15 Pro
  Max, 6.765 mm / f/1.78, real EXIF), within 72 seconds of each other, confirming a single
  stationary-object capture session.
- **Registration results** (RANSAC homography, View N -> View 1; full precision and every
  intermediate count in `docs/SFM_CALCULATIONS.md` Section 6):

  | View            | Inliers / retained matches | Mean reprojection error (inliers) |
  | ---------------- | --------------------------- | ----------------------------------- |
  | View 2 -> View 1 | 53 / 112 (47.3%)             | 1.546 px                            |
  | View 3 -> View 1 | 7 / 30 (23.3%)                | 0.877 px                            |
  | View 4 -> View 1 | 31 / 72 (43.1%)               | 1.285 px                            |

  View 3 registered with fewer matches than View 2/4 after diagnosis showed many of its
  candidate ORB matches were ambiguous (the cover's repeated/self-similar title lettering
  produced locally-plausible but globally-inconsistent correspondences at that viewing angle);
  switching to Lowe's ratio test (documented in `scripts/process_sfm_experiment.py`) recovered
  a genuinely consistent, low-error homography from the correspondences that remained. See
  `docs/EXPERIMENTAL_RESULTS.md` Section 9 for the full diagnosis and before/after numbers.
- **Boundary reconstruction**: the four real, manually identified boundary corners in View 1
  and the three homography-registered boundaries from Views 2-4 agree closely (see the overlay
  at `results/sfm/reference_boundary_reconstruction.jpg`); a normalized top-down rectification
  of View 1's cover is at `results/sfm/view_1_top_down_rectified.jpg`.

## 1. What the assignment asks for

IMPLEMENTATION_PLAN.md Sections 13-14: an example of Structure From Motion using four
different camera viewpoints of one stationary object, "for simplicity, use a flat planar
object as permitted by the assignment." For each view, record the image and (where actually
available) camera/device information, approximate position and orientation, and distance to
the object.

## 2. Why a planar object simplifies this considerably

General Structure From Motion (arbitrary 3D scenes, multiple views, unknown camera poses)
requires estimating a fundamental/essential matrix, recovering relative camera pose, and
triangulating 3D points from 2D observations - substantially more machinery, and the
assignment explicitly does not require it for this question. Because the assignment's object
is flat, the entire object can be modeled as living on a single plane (`Z = 0` in the object's
own coordinate frame). For a plane, the mapping between any two camera views of that same
plane collapses to a single 3x3 **homography** - a purely 2D projective transformation between
the two images (IMPLEMENTATION_PLAN.md Section 16; derivation in `docs/CAMERA_GEOMETRY.md`
Section 4). This is a standard, well-established simplification, not an ad hoc shortcut - see
Hartley and Zisserman [1], Chapter 13, "Scene planes and homographies."

## 3. What this implementation actually computes

Given four images of the same flat object from four camera positions:

1. Pick one view as the **reference** view.
2. For each other view, find point correspondences with the reference (`module5_6.features`,
   automatic via ORB, or manually supplied per IMPLEMENTATION_PLAN.md Section 17).
3. Estimate the planar homography that maps that view's pixel coordinates into the reference
   view's pixel coordinates (`module5_6.homography`), using RANSAC to identify and exclude
   outlier correspondences.
4. Validate the estimate with per-point and mean reprojection error (IMPLEMENTATION_PLAN.md
   Section 19).
5. If the object's boundary/corner points are known in a given view, map them into the
   reference frame too, and combine the registered boundaries from all views into one
   consensus estimate of the object's boundary in the reference frame
   (IMPLEMENTATION_PLAN.md Section 20) - `module5_6.sfm.register_views`.

## 4. Terminology - what this is not

This is a **2D planar homography registration**, not a dense or full 3D Structure-from-Motion
reconstruction. In particular, this implementation (Phase 5):

- does **not** recover a 3D point cloud or a 3D model of the object;
- does **not** estimate the 3D rotation/translation between camera poses (no fundamental or
  essential matrix, no triangulation);
- does **not** produce a depth map.

IMPLEMENTATION_PLAN.md Section 21 describes an *optional* full two-view epipolar pipeline
(fundamental matrix -> essential matrix -> pose recovery -> triangulation) that would
produce an actual 3D reconstruction; it is explicitly an enhancement, not a requirement, and
is not implemented in this phase. Referring to the planar-homography result produced here as
"3D reconstruction" would misrepresent what was actually computed, so this documentation and
the corresponding web page avoid that phrasing throughout.

## 5. Real-experiment requirements (completed in Phase 6)

With the four real images in place (IMPLEMENTATION_PLAN.md Section 14), Phase 6 did all of the
following, via `scripts/process_sfm_experiment.py`:

- recorded each view's filename, image dimensions, and whatever camera/device, focal length,
  and (user-recorded) approximate position/orientation/distance-to-object information is
  actually available (`module5_6.camera.ViewMetadata`, `results/sfm/sfm_summary.json` ->
  `views`) - every field the real data does not support stays `None`, never a plausible-looking
  guess (see `docs/CAMERA_GEOMETRY.md` Section 5 and `module5_6.camera`'s docstring);
- chose View 1 as the reference view and registered the other three
  (`module5_6.sfm.register_views`);
- reported real per-point and mean reprojection errors and the recovered/registered boundary
  (Section 0 above, `docs/SFM_CALCULATIONS.md` Section 6);
- includes the mathematical workout required by IMPLEMENTATION_PLAN.md Section 22, using one
  actual matched point from the real images (`docs/SFM_CALCULATIONS.md` Section 7) - the
  earlier Section 4 worked example above remains as an explanatory, invented-number example
  only and is never presented as this real result.

## References

1. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.
   Cambridge University Press, 2004.
2. See `docs/CAMERA_GEOMETRY.md` and `docs/SFM_CALCULATIONS.md` for the additional references
   (ORB, OpenCV `findHomography`) used in the specific mathematical/implementation derivations.
