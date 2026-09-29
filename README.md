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
all wired into the Motion Tracking page. Bilinear interpolation, structure from motion, and all
empirical results remain scheduled for later approved phases. No results, pixel coordinates,
camera parameters, or SfM measurements are fabricated; the two assignment videos and their
required two-consecutive-frame pixel-location validation remain PENDING USER EXPERIMENT until
supplied.

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
  quality signals from whatever video is uploaded, not the assignment's required manual
  pixel-location validation (a later phase).
- Bilinear Interpolation & Theory, Structure From Motion, and Experiments & Results remain
  pending-safe placeholders for later approved phases.

## Run tests

    python -m pytest -q

Tests cover image validation, BGR decoding, foundation types, video metadata/sample-range/
frame-reading behavior against small synthetic video fixtures, Farneback optical-flow
correctness against a known synthetic pixel translation, flow visualization, Shi-Tomasi feature
detection, Lucas-Kanade tracking correctness against a known synthetic pixel translation,
valid/invalid track filtering, forward-backward validation, multi-frame trajectory building,
tracking visualization, and the dashboard-compatible page-provider contract. They do not count
as experimental validation - that requires the actual assignment videos (see "Data and video
requirements").

## Planned reproduction workflow

Later phases will add scripts such as:

    python scripts/process_optical_flow.py
    python scripts/validate_tracking.py
    python scripts/run_sfm.py
    python scripts/run_experiments.py

User videos will be supplied under `data/videos/video_1/` and `data/videos/video_2/`, and the
four SfM viewpoint images under `data/sfm/view_1/` through `data/sfm/view_4/`, or through the
app. Large video files and user-collected images are not committed by default. Until real
videos and four-view images exist, tracking errors, camera parameters, and SfM
reprojection/boundary results remain pending user data collection.

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
      SfM/theory modules (added in later phases)
      webapp/             PageSpec provider and Streamlit UI
    data/                 user videos and four-view SfM images
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
