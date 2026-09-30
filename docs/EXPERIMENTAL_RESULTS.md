# Experimental Results

Consolidated report-ready results for Question 1's real-video experiments: optical-flow
evidence for each video, and the two-consecutive-frame pixel-location tracking validation.
Structure-from-motion experimental results (Question 2) are a later phase and are not covered
here.

## Status

**PENDING USER EXPERIMENT - the two required assignment videos have not been supplied.**

Checked with `module5_6.experiment.find_supplied_video`, which looks for any real (non-dotfile)
file under a directory and is what the Experiments & Results web page uses for the same check:

| Video   | Directory              | Status                     |
| ------- | ----------------------- | -------------------------- |
| Video 1 | `data/videos/video_1/`  | PENDING USER EXPERIMENT    |
| Video 2 | `data/videos/video_2/`  | PENDING USER EXPERIMENT    |

Both directories currently contain only a `.gitkeep` placeholder (see `git ls-files
data/videos/`). No optical-flow evidence, tracked pixel coordinate, predicted or observed
Frame 2 location, or pixel error below has been measured from a real video. Nothing in this
document is fabricated to fill that gap.

## 1. Video summaries

| Field | Video 1 | Video 2 |
| --- | --- | --- |
| Source file | Pending | Pending |
| Resolution | Pending | Pending |
| FPS | Pending | Pending |
| Sample interval used | Pending (>= 30 s required) | Pending (>= 30 s required) |
| Motion content | Pending | Pending |

Once each video is supplied, run:

```bash
python scripts/process_optical_flow.py --video path/to/video_1.mp4 --video-id video_1
```

which fills in this table's fields from `results/optical_flow/video_1/
video_1_optical_flow_summary.json` (and the analogous file for Video 2), and produces the
optical-flow visualization video referenced by Question 1's "compute optical flow and
visualize the same as a video" requirement. The Experiments & Results page reads these summary
files automatically and displays them once they exist.

## 2. Information inferred from optical flow

Pending. IMPLEMENTATION_PLAN.md Section 6 requires this discussion to be tied to specific
screenshots, vectors, or measurements from the actual selected videos (direction of motion,
relative motion magnitude, moving vs. static regions, trajectories, camera motion, and so on).
None of that discussion can be written honestly before the real videos and their optical-flow
evidence exist.

## 3. Two-consecutive-frame pixel-location tracking validation

Procedure, coordinate convention, point-selection method, and the error formula are documented
in `docs/TRACKING_VALIDATION.md`. Results table (IMPLEMENTATION_PLAN.md Section 12):

| Video   | Point | Frame 1 | Predicted Frame 2 | Observed Frame 2 | Pixel Error |
| ------- | ----- | ------- | ------------------ | ----------------- | ----------- |
| Video 1 | P1    | Pending | Pending             | Pending            | Pending     |
| Video 1 | P2    | Pending | Pending             | Pending            | Pending     |
| Video 2 | P1    | Pending | Pending             | Pending            | Pending     |
| Video 2 | P2    | Pending | Pending             | Pending            | Pending     |

Reproduction: `docs/TRACKING_VALIDATION.md` Section 4 (script workflow) or the Motion Tracking
web page's manual-validation section. Completed records can be uploaded on the Experiments &
Results page to regenerate this table automatically, or exported with
`module5_6.experiment.write_records_csv`.

## 4. Discussion

Pending real results.

## 5. Reproducibility notes

- `data/experiment_manifest.json` records per-video status (`pending_user_experiment` /
  `available` / `processed`), path, and basic metadata once known; its schema matches
  `module5_6.experiment.default_experiment_manifest`.
- All frame indices, coordinates, and errors in a completed experiment are traceable to a
  specific video file, frame pair, and (for the observed coordinate) an `observation_method`
  string recorded on the corresponding `TrackingValidationRecord`.
- Automated tests (`tests/test_video.py`, `tests/test_optical_flow.py`, `tests/test_tracking.py`,
  `tests/test_experiment.py`) verify the underlying code against synthetic fixtures with known
  answers; they are software-correctness evidence, not a substitute for this document's
  real-video results.
