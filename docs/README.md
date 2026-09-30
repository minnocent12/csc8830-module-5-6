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
  planar simplification applies, what this implementation computes, the important terminology
  distinction from dense/full 3D reconstruction, and (Section 0) a summary of the completed
  real four-view experiment.
- `CAMERA_GEOMETRY.md` - the pinhole camera model, homogeneous coordinates, the intrinsic
  matrix `K` and extrinsic `[R|t]`, the planar-scene homography relation
  `H = K(R - t n^T/d) K^-1`, and (Section 5) exactly which of those quantities are known vs.
  genuinely unknown for the real experiment.
- `SFM_CALCULATIONS.md` - the Direct Linear Transform homography estimation, reprojection
  validation, boundary registration across views, a deterministic synthetic worked example
  (Section 4, explanatory only), and (Sections 6-7) the real four-view registration results
  and worked example using actual image data.
- `EXPERIMENTAL_RESULTS.md` Section 9 - the consolidated real SfM results write-up (object,
  camera-position table, feature/match/inlier counts, homographies, reprojection error,
  boundary reconstruction).

**COMPLETE.** The four real `data/sfm/view_1..4/` images were captured and processed
end-to-end by `scripts/process_sfm_experiment.py`; every number in the documents above comes
from that run (`results/sfm/sfm_summary.json`). The reusable software foundation
(`module5_6.features`, `homography`, `camera`, `geometry`, `sfm`) was built and tested against
synthetic fixtures first, then run on the real images.

## Documents outside `docs/`

- `../README.md` - setup, running the app (standalone and shared-dashboard), architecture, and
  current status.
- `../data/README.md` - what belongs under `data/`, what is gitignored vs. committed, and the
  `experiment_manifest.json` schema.
- `../DEPLOYMENT.md` (root `Assignments/` level) - how this module is wired into the public
  Streamlit Community Cloud dashboard.

Report notes and the demonstration-video checklist for the final PDF/video submission are a
later phase (see `IMPLEMENTATION_PLAN.md` for the full planned document list) - not started
yet, and out of scope for the web-application-completion phase.

No empirical claims belong in these documents until corresponding experiments have actually
run. Worked numerical examples and any synthetic tracking/flow/homography examples referenced
from these documents are mathematical or software-verification examples, not assignment
experimental evidence.
