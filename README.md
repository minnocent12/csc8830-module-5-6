# CSc 8830 Module 5-6 - Optical Flow, Motion Tracking, and Structure From Motion

**Public web app (all modules):** <https://csc8830-dashboard-minnocent1.streamlit.app>

This is the independent Module 5-6 repository for Georgia State University CSc 8830 Computer
Vision. The assignment covers two questions:

- **Question 1** - optical flow and motion tracking on two user-captured videos, including a
  from-fundamentals derivation of the motion-tracking equations, a bilinear-interpolation
  derivation, and pixel-level tracking validation on two consecutive frames from each video.
- **Question 2** - structure from motion from four viewpoints of a flat/2D planar object,
  recovering the object's boundary with the supporting mathematical workouts, camera
  positions, and camera parameters.

## Current status

Phase 0 establishes the independent package, safe image IO, data/result directories, the
page-provider contract, and pending-safe Streamlit navigation. Optical flow, motion tracking,
bilinear interpolation, structure from motion, and all empirical results are scheduled for
later approved phases. No results, pixel coordinates, camera parameters, or SfM measurements
are fabricated.

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

The current foundation exposes five pending-safe pages:

- Optical Flow
- Motion Tracking
- Bilinear Interpolation & Theory
- Structure From Motion
- Experiments & Results

## Run tests

    python -m pytest -q

Tests cover image validation, BGR decoding, foundation types, and the dashboard-compatible
page-provider contract. They do not count as experimental validation.

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
      core CV, theory, and SfM modules (added in later phases)
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
`../DEPLOYMENT.md`) is a later step, not part of this foundation phase. The standalone
Module 5-6 app remains the recommended grading path in the meantime.

## Limitations and integrity notes

- Optical-flow accuracy is scene- and motion-dependent; results will document the actual
  videos used.
- The four-view structure-from-motion demonstration assumes a flat/2D planar object, per the
  assignment's stated simplification.
- No claim of tracking accuracy, tracking error, or SfM reprojection accuracy will be made
  before the corresponding real experiment has actually run.

## GitHub repository

<!-- Filled in once the independent Module 5-6 GitHub repository is created and pushed. -->
