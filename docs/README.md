# Module 5-6 documentation

## Question 1 - Optical flow, tracking, and bilinear interpolation

- `OPTICAL_FLOW_THEORY.md` - brightness constancy, the first-order Taylor expansion, the
  optical-flow constraint equation, and the aperture problem.
- `MOTION_TRACKING_DERIVATION.md` - the Lucas-Kanade overdetermined system and least-squares
  solution, the two-frame tracking problem, and how that derivation relates to the pyramidal
  OpenCV implementation used in Phase 2.
- `BILINEAR_INTERPOLATION.md` - the bilinear-interpolation derivation from two sequential 1D
  linear interpolations, plus a complete deterministic numerical worked example.
- `TRACKING_VALIDATION.md` - the professor-required two-consecutive-frame manual
  pixel-location validation: procedure, coordinate convention, point-selection method, the
  Euclidean pixel-error formula, results, and limitations. **Complete** for one validation
  point per real assignment video (`IMG_7272.MOV`, `IMG_7275.MOV`).
- `EXPERIMENTAL_RESULTS.md` - the consolidated results write-up (video summaries, inferred
  optical-flow information, the validation table). **Complete** for both real assignment
  videos.

## Question 2 - Structure From Motion (planar homography registration)

- `STRUCTURE_FROM_MOTION_THEORY.md` - what the four-view planar SfM task asks for, why the
  planar simplification applies, what this implementation computes, and the important
  terminology distinction from dense/full 3D reconstruction.
- `CAMERA_GEOMETRY.md` - the pinhole camera model, homogeneous coordinates, the intrinsic
  matrix `K` and extrinsic `[R|t]`, and the planar-scene homography relation
  `H = K(R - t n^T/d) K^-1`.
- `SFM_CALCULATIONS.md` - the Direct Linear Transform homography estimation, reprojection
  validation, boundary registration across views, and a complete deterministic numerical
  worked example.

**PENDING USER EXPERIMENT** until the four real `data/sfm/view_1..4/` images are supplied - see
`STRUCTURE_FROM_MOTION_THEORY.md`'s Status section. Phase 5 builds and tests the reusable
software foundation (`module5_6.features`, `homography`, `camera`, `geometry`, `sfm`) only,
against synthetic fixtures.

Later approved phases will add report notes and the demonstration-video checklist (see
`IMPLEMENTATION_PLAN.md` for the full planned document list).

No empirical claims belong in these documents until corresponding experiments have actually
run. Worked numerical examples and any synthetic tracking/flow/homography examples referenced
from these documents are mathematical or software-verification examples, not assignment
experimental evidence.
