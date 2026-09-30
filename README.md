# CSc 8830 Module 5-6 - Optical Flow, Motion Tracking, and Structure From Motion

**Public dashboard (Modules 2-4 today):** <https://csc8830-dashboard-minnocent1.streamlit.app>
Module 5-6 is not yet wired into the public dashboard (see "Optional shared dashboard" below);
until then, run the standalone app locally with `streamlit run app.py`.

This is the independent Module 5-6 repository for Georgia State University CSc 8830 Computer
Vision. The assignment covers two questions:

- **Question 1** - optical flow and motion tracking on two user-captured videos, including a
  from-fundamentals derivation of the motion-tracking equations, a bilinear-interpolation
  derivation, and pixel-level tracking validation on two consecutive frames from each video.
- **Question 2** - structure from motion from four viewpoints of a flat/2D planar object,
  recovering the object's boundary with the supporting mathematical workouts, camera
  positions, and camera parameters.

## Current status

Phase 0 established the independent package, safe image IO, data/result directories, the
page-provider contract, and pending-safe Streamlit navigation. Phase 1 adds video loading and
validation, video metadata, configurable sample-interval extraction (with the assignment's
30-second minimum enforced by default), consecutive-frame reading, grayscale conversion, dense
Farneback optical flow, magnitude/direction computation, HSV-color and arrow-overlay
visualization, and processed optical-flow video export - all wired into the Optical Flow page.
Phase 2 adds Shi-Tomasi feature detection, pyramidal Lucas-Kanade tracking between consecutive
frames, valid/invalid track filtering from OpenCV's status output, forward-backward tracking
validation, displacement vectors/magnitude, and multi-frame trajectory/track-history support -
all wired into the Motion Tracking page. Phase 3 adds a from-scratch bilinear-interpolation
implementation (independent of OpenCV, cross-validated against it), the report-ready
brightness-constancy/optical-flow-constraint/aperture-problem/Lucas-Kanade derivations, and the
bilinear-interpolation derivation with a numerical worked example - all presented, with an
interactive demonstration, on the Bilinear Interpolation & Theory page. Phase 4 adds the
professor-required two-consecutive-frame manual pixel-location validation infrastructure:
`scripts/process_optical_flow.py` and `scripts/validate_tracking.py`, the
`module5_6.experiment` validation-record/manifest/overlay helpers, a manual-validation section
on the Motion Tracking page, and a fully implemented Experiments & Results page. **The two
required assignment videos have now been supplied and processed**
(`data/videos/video_1/IMG_7272.MOV`, `data/videos/video_2/IMG_7275.MOV`, each gitignored
locally-only, not committed): a 30-second optical-flow sample and one manually-validated
tracking point were completed for each - pixel errors 4.628 px (video_1) and 8.408 px
(video_2). Full results, evidence figures, and reproduction commands:
`docs/EXPERIMENTAL_RESULTS.md` and `docs/TRACKING_VALIDATION.md`. Phase 5 adds the reusable
planar Structure-From-Motion foundation for Question 2: ORB feature detection/matching
(`module5_6.features`), homography estimation/reprojection with RANSAC outlier rejection
(`module5_6.homography`), camera/view metadata with no fabricated physical parameters
(`module5_6.camera`), homogeneous-coordinate/boundary-transform helpers (`module5_6.geometry`),
and a multi-view registration coordinator (`module5_6.sfm`) - all wired into the Structure From
Motion page. **The four required assignment viewpoints have not been supplied yet**
(`data/sfm/view_1/` through `view_4/` contain only `.gitkeep`), so this is tested foundation
only, not the real four-view experiment - see `docs/STRUCTURE_FROM_MOTION_THEORY.md`. No
results, pixel coordinates, camera parameters, or SfM measurements are fabricated.

## Setup

Python 3.10 or newer is required.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -e ".[dev]"

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Run the app

From this repository root:

    streamlit run app.py

The app exposes five pages:

- **Optical Flow** - upload a video (mp4/avi/mov/mkv/m4v) to compute dense Farneback optical
  flow over a configurable sample interval. Controls: start time, sample duration (30-second
  assignment minimum enforced by default, with an explicit opt-out for quick previews),
  visualization mode (HSV color or arrow overlay), vector spacing, arrow magnitude threshold,
  and a frame cap that bounds interactive compute time. Outputs: the sample's start/end
  frames, a representative HSV-color and arrow-overlay frame pair, a downloadable/playable
  optical-flow video, and live magnitude/direction statistics. These statistics are real
  computed values from whatever video is uploaded, but they are exploratory tooling, not the
  assignment's required two-frame pixel-tracking validation (a later phase).
- **Motion Tracking** - upload a video to detect Shi-Tomasi features on a start frame and track
  them with pyramidal Lucas-Kanade into the next consecutive frame (the assignment's two-frame
  tracking problem) and across a longer loaded window for track history. Controls: start time,
  track-history length, Shi-Tomasi settings (max corners, quality level, min distance, block
  size, Harris toggle), Lucas-Kanade settings (window size, pyramid levels, iterations,
  epsilon), and an optional forward-backward validation threshold. Outputs: Frame 1 with
  detected features, Frame 2 with tracked features, a displacement-vector overlay, a
  point-coordinate/displacement/error table, live tracking statistics, and a trajectory overlay
  across the loaded frames. OpenCV's per-point and forward-backward errors are algorithmic
  quality signals from whatever video is uploaded - distinct from the **two-frame
  pixel-location validation (manual, professor-required)** section below the table, which
  requires the user to visually inspect a zoomed Frame 2 crop, enter the actual observed
  coordinate (defaulted to the Frame 1 location, never to the prediction, so it cannot be
  accepted unedited), and explicitly confirm the observation before a real pixel error is
  computed, shown, and made downloadable as a JSON record.
- **Bilinear Interpolation & Theory** - presents the brightness-constancy assumption, the
  first-order Taylor expansion and optical-flow constraint equation, the aperture problem, the
  Lucas-Kanade overdetermined system and least-squares solution (and its relationship to the
  Phase 2 OpenCV implementation), and the bilinear-interpolation derivation from two sequential
  1D linear interpolations. Includes an interactive demonstration: configurable four
  neighboring pixel intensities and fractional coordinates, live interpolation weights and
  result, and a schematic diagram, computed by the same `module5_6.interpolation` functions
  the tests use. Full write-ups: `docs/OPTICAL_FLOW_THEORY.md`,
  `docs/MOTION_TRACKING_DERIVATION.md`, `docs/BILINEAR_INTERPOLATION.md`.
- **Structure From Motion** - upload up to four images of one flat/2D planar object. Controls:
  per-view optional camera/device metadata (device, focal length, distance, orientation, notes
  - never inferred from image size), a reference-view selector, ORB/homography settings
  (max features; RANSAC vs. all-points; RANSAC threshold), and optional per-view boundary
  corner coordinates. Outputs: detected ORB features on the reference view, matched-point
  overlays for each other view, inlier counts, mean reprojection error (over inliers), the
  estimated homography matrix, and (when boundary corners were entered) the registered
  boundary overlaid on the reference view. This is a 2D planar homography registration, not a
  dense/full 3D reconstruction - see `docs/STRUCTURE_FROM_MOTION_THEORY.md`. Computed live from
  whatever images are uploaded; the real four-view assignment experiment remains
  PENDING USER EXPERIMENT until `data/sfm/view_1/` through `view_4/` are supplied.
- **Experiments & Results** - shows per-video status (whether a real video has been supplied
  under `data/videos/video_1|video_2/`), any generated optical-flow evidence summary, and a
  consolidated two-consecutive-frame pixel-location validation table. Completed records under
  `results/tracking/<video_id>/*_record.json` (produced by `scripts/validate_tracking.py` or
  the Motion Tracking page) are loaded automatically; additional record JSON files can be
  uploaded too. Currently shows both videos as processed and both P1 records complete (pixel
  errors 4.628 px and 8.408 px - see `docs/EXPERIMENTAL_RESULTS.md`); the table shape still
  falls back to IMPLEMENTATION_PLAN.md Section 12's `PENDING USER EXPERIMENT` rows whenever no
  records are found. Also reports whether any real `data/sfm/view_N/` image has been supplied.

## Run tests

    python -m pytest -q

Tests cover image validation, BGR decoding, foundation types, video metadata/sample-range/
frame-reading behavior against small synthetic video fixtures, Farneback optical-flow
correctness against a known synthetic pixel translation, flow visualization, Shi-Tomasi feature
detection, Lucas-Kanade tracking correctness against a known synthetic pixel translation,
valid/invalid track filtering, forward-backward validation, multi-frame trajectory building,
tracking visualization, bilinear-interpolation weights/values against hand-worked examples and
against an OpenCV `cv2.remap` cross-check, the Phase 4 experiment-record/manifest/overlay
helpers (`module5_6.experiment`) including the pixel-error formula and JSON/CSV
round-tripping, ORB feature detection/matching against a known synthetic pixel translation
(`module5_6.features`), homography estimation (identity/translation/rotation-scale/known
projective transform/exact four-point/noisy correspondences/RANSAC outlier rejection/degenerate
rejection) and reprojection error against hand-worked formulas (`module5_6.homography`),
camera/view metadata validation with no fabricated physical-parameter defaults
(`module5_6.camera`), homogeneous-coordinate and boundary-transform geometry
(`module5_6.geometry`), end-to-end synthetic view registration recovering a known homography
from warped images (`module5_6.sfm`), and the dashboard-compatible page-provider contract.
They do not count as experimental validation - that requires the actual assignment videos and
four-view images (see "Data and video
requirements").

## Reproduction workflow

Once a real video is supplied, generate its required evidence with:

    python scripts/process_optical_flow.py --video path/to/video_1.mp4 --video-id video_1
    python scripts/validate_tracking.py prepare --video path/to/video_1.mp4 --video-id video_1 --frame1 <N>
    python scripts/validate_tracking.py record --record results/tracking/video_1/P1_record.json \
        --observed-x <X> --observed-y <Y> --video path/to/video_1.mp4

See `docs/TRACKING_VALIDATION.md` for the full procedure, coordinate convention, and the
distinction between the algorithmic and manually-observed pixel errors. `scripts/run_sfm.py`
and `scripts/run_experiments.py` (structure-from-motion and cross-phase report figures) remain
later-phase additions.

User videos are supplied under `data/videos/video_1/` and `data/videos/video_2/`, and the four
SfM viewpoint images under `data/sfm/view_1/` through `data/sfm/view_4/`, or through the app.
Large video files and user-collected images are not committed by default. Until real videos and
four-view images exist, tracking errors, camera parameters, and SfM reprojection/boundary
results remain pending user data collection.

## Data and video requirements

- Two videos, each with at least a 30-second sample containing visible motion and sufficient
  texture/features for tracking.
- Four images of a single flat/2D planar object of choice, captured from four different camera
  viewpoints, with recorded camera position/orientation and, where available, intrinsic
  parameters.

See `data/README.md` for the exact expected layout.

## Architecture

    app.py
    src/module5_6/
      types.py            shared dataclasses (image/video metadata, flow/experiment status)
      io_utils.py         safe BGR image IO and grayscale conversion
      video.py            video loading, metadata, sample-interval extraction, frame reading,
                           and optical-flow video export
      optical_flow.py     pure Farneback dense-flow computation and visualization
      tracking.py         Shi-Tomasi detection, pyramidal Lucas-Kanade tracking, forward-
                           backward validation, trajectories, and tracking visualization
      interpolation.py    from-scratch bilinear interpolation (four-neighbor lookup, weights,
                           full breakdown), independent of OpenCV
      experiment.py       Phase 4 validation-record/manifest/overlay helpers implementing the
                           predicted-vs-observed pixel-error formula
      features.py         ORB feature detection and descriptor matching (automatic planar
                           point correspondences)
      homography.py       planar homography estimation (RANSAC/LMedS/all-points), point
                           transform, and per-point/mean reprojection error
      camera.py           per-view camera/device metadata - every physical parameter defaults
                           to None, never fabricated or inferred from image size
      geometry.py         homogeneous-coordinate conversion and homography-based point/
                           boundary transforms
      sfm.py              coordinates features/homography/geometry into multi-view planar
                           registration (not dense/full 3D reconstruction)
      webapp/             PageSpec provider and Streamlit UI
    scripts/
      process_optical_flow.py   generate optical-flow evidence for one real video
      validate_tracking.py      two-step manual pixel-location validation (prepare / record)
    data/                 user videos, four-view SfM images, and experiment_manifest.json
    results/              derived optical-flow/tracking/SfM outputs and metrics
    docs/                 theory, derivations, results, report, and demo notes
    tests/                deterministic automated tests

Core processing will remain importable without Streamlit and will be reused by scripts and the
web app. Module 5-6 has no runtime dependency on Module 2, Module 3, or Module 4.

## Optional shared dashboard

When multiple independent module repositories are placed beside one another, a host can mount
Module 5-6 with:

    from module5_6.webapp.pages import get_pages

The host should add `Module_5-6/src` to its import path and adapt page objects by their
`module_label`, `page_label`, `order`, and `render` attributes. Wiring Module 5-6 into the
course root dashboard and the public Streamlit Community Cloud deployment (see
`../DEPLOYMENT.md`) is a later step, not part of these early phases. The standalone
Module 5-6 app remains the recommended grading path in the meantime.

## Limitations and integrity notes

- Optical-flow accuracy is scene- and motion-dependent; results will document the actual
  videos used.
- The four-view structure-from-motion demonstration assumes a flat/2D planar object, per the
  assignment's stated simplification.
- No claim of tracking accuracy, tracking error, or SfM reprojection accuracy will be made
  before the corresponding real experiment has actually run.

## GitHub repository

<https://github.com/minnocent12/csc8830-module-5-6>
