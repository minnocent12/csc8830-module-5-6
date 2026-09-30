# Structure From Motion Theory (Question 2)

This document gives the conceptual overview of Question 2's four-view planar Structure From
Motion task and states precisely what this implementation does and does not claim. The
underlying camera/homography mathematics is in `docs/CAMERA_GEOMETRY.md`; the concrete
calculation workflow and a worked numerical example are in `docs/SFM_CALCULATIONS.md`.

**Status: PENDING USER EXPERIMENT.** `data/sfm/view_1/` through `view_4/` contain no real
images yet (only `.gitkeep` placeholders - verified with
`module5_6.experiment.find_supplied_video`-style directory checks). This phase (Phase 5)
builds and tests the reusable software foundation only, against synthetic fixtures. No camera
position, focal length, correspondence, or reprojection error in this document is invented.

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

## 5. Real-experiment requirements (later phase)

Once four real images exist (IMPLEMENTATION_PLAN.md Section 14):

- record each view's filename, image dimensions, and whatever camera/device, focal length,
  intrinsic, approximate position/orientation, and distance-to-object information is actually
  available (`module5_6.camera.ViewMetadata`) - fields the user cannot supply must stay
  `None`, never a plausible-looking guess (see `docs/CAMERA_GEOMETRY.md` Section 3 and
  `module5_6.camera`'s docstring);
- choose a reference view and register the other three (`module5_6.sfm.register_views`);
- report real per-point and mean reprojection errors and the recovered/registered boundary;
- include the mathematical workouts required by IMPLEMENTATION_PLAN.md Section 22 as typed,
  digitized work in the final report, following the worked example format in
  `docs/SFM_CALCULATIONS.md`.

## References

1. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.
   Cambridge University Press, 2004.
2. See `docs/CAMERA_GEOMETRY.md` and `docs/SFM_CALCULATIONS.md` for the additional references
   (ORB, OpenCV `findHomography`) used in the specific mathematical/implementation derivations.
