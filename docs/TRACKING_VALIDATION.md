# Two-Consecutive-Frame Pixel-Location Tracking Validation

This document specifies the procedure, coordinate convention, point-selection method, error
formula, and limitations for the professor-required validation in
IMPLEMENTATION_PLAN.md Section 12 / the Module 5-6 assignment questions:

> "Pick two consecutive frames from each of the videos and validate the theoretical result of
> tracking with actual pixel locations."

**Status: COMPLETE for one validation point per video.** The two real assignment videos
(`data/videos/video_1/IMG_7272.MOV`, `data/videos/video_2/IMG_7275.MOV`) have been supplied and
processed; Section 6 below reports the real, manually-measured results. A second point per
video was attempted but not included - see Section 6's note on that. No video identity, frame
number, pixel coordinate, or pixel error in Section 6 is invented; each was produced by the
procedure in Section 4, verified with `module5_6.experiment.find_supplied_video` and the
Experiments & Results page, which loads these same records automatically from
`results/tracking/`.

## 1. What this validation is, and what it is not

For a point `P = (x1, y1)` selected in Frame 1, this validation compares two things that must
come from genuinely different sources:

- the **predicted** location `(predicted_x2, predicted_y2)` in Frame 2 - the *theoretical/
  algorithmic* result, computed by running Phase 2's pyramidal Lucas-Kanade tracker
  (`module5_6.tracking.track_points`) on that one point; and
- the **observed** location `(observed_x, observed_y)` in Frame 2 - a real human measurement,
  obtained by actually looking at the Frame 2 image and determining where the point is.

The Euclidean pixel error is:

```
e = sqrt((predicted_x2 - observed_x)^2 + (predicted_y2 - observed_y)^2)
```

matching IMPLEMENTATION_PLAN.md Section 12 exactly (there written as
`e = sqrt((x̂2-x2)^2 + (ŷ2-y2)^2)`), and implemented as
`module5_6.experiment.TrackingValidationRecord.pixel_error`.

**This is distinct from, and must never be confused with:**

- OpenCV's own per-point Lucas-Kanade tracking error (`TrackedPoints.error`, Phase 2) - an
  algorithmic measure of how well the tracker's internal model fit, not a comparison to a real
  observation;
- forward-backward consistency error (`ForwardBackwardResult.fb_error`, Phase 2) - a
  self-consistency check (track forward then backward, measure drift), also without ever
  looking at what is actually in Frame 2;
- any synthetic/known-translation error used only in automated software tests
  (`tests/test_tracking.py`, `tests/test_optical_flow.py`, `tests/test_experiment.py`) - these
  verify the code is implemented correctly, using an artificially constructed shift with a
  known answer, and are never presented as real video evidence.

A `TrackingValidationRecord` with `observed_x`/`observed_y` left as `None` has
`pixel_error is None` and `status = "awaiting_manual_observation"` - no number is ever
substituted or estimated for a missing observation.

## 2. Coordinate convention

Identical to the rest of this codebase (`module5_6.video`, `optical_flow`, `tracking`,
`interpolation`): `x` increases rightward (column index), `y` increases downward (row index).
Frame indices are 0-based, counted from the start of the sample interval read from the video.

## 3. Point-selection method

**Frame 1 point.** Either:

- auto-detected by Shi-Tomasi (`module5_6.tracking.detect_features`) and picked by the user
  from the ranked candidates (the default in both the web page and
  `scripts/validate_tracking.py prepare`), or
- entered explicitly as `(x1, y1)` (`--x1`/`--y1` on the script, or typed into the web page).

**Predicted Frame 2 point.** Always computed by `module5_6.experiment.predict_frame2_location`,
which wraps Phase 2's `track_points` for that single point. If OpenCV's own status output
marks the point as not reliably tracked, the workflow prints/shows an explicit warning rather
than silently trusting the prediction.

**Observed Frame 2 point - the part that must be a real measurement.** Two supported,
reproducible workflows:

1. **Script workflow.** `scripts/validate_tracking.py prepare` writes a Frame 2 evidence image
   with the predicted location marked. A human opens that image (or the original Frame 2
   frame) in an image viewer capable of reporting pixel coordinates under the cursor, reads
   off where the point actually is, and passes that as `--observed-x`/`--observed-y` to
   `scripts/validate_tracking.py record`.
2. **Web page workflow.** The Motion Tracking page's "Two-frame pixel-location validation"
   section shows a zoomed crop of Frame 2 around the predicted location. The user inspects it,
   types the actual observed `(x, y)` into the two number inputs (which default to the Frame 1
   coordinate, *not* the prediction, specifically so an unedited default cannot silently pass
   as an observation), and must check an explicit confirmation box
   ("I have visually inspected Frame 2 ... not copied from the prediction") before a pixel
   error is computed or shown at all.

In both workflows, `observation_method` is recorded on the saved record (e.g. "manual pixel
inspection", "Streamlit Motion Tracking page - manual visual inspection") so the measurement
stays reproducible and auditable.

## 4. Reproducible procedure (once a real video exists)

For each of the two required videos:

```bash
# 1. Generate optical-flow evidence for the video (Question 1's "visualize as a video").
python scripts/process_optical_flow.py --video path/to/video_1.mp4 --video-id video_1

# 2. Prepare a validation record for one point on a chosen consecutive frame pair.
python scripts/validate_tracking.py prepare \
    --video path/to/video_1.mp4 --video-id video_1 --frame1 <N> --point-label P1

# 3. Visually inspect results/tracking/video_1/P1_frame2_predicted.png (or the original
#    Frame 2 frame) and determine the actual observed pixel location.

# 4. Record that real observation and compute the pixel error.
python scripts/validate_tracking.py record \
    --record results/tracking/video_1/P1_record.json \
    --observed-x <X> --observed-y <Y> --video path/to/video_1.mp4
```

Repeat for at least one more point per video (the required table in Section 6 below has two
points per video) and for the second video. The web page's manual-validation section on the
Motion Tracking page offers the same workflow interactively, with a JSON download for each
completed record.

Completed records can be consolidated (as CSV or as the report table) with
`module5_6.experiment.write_records_csv` / `records_to_table`, or uploaded on the
Experiments & Results page.

## 5. Limitations

- **Manual observation precision.** A human reading a pixel coordinate off an image or a
  zoomed crop has limited precision (typically a few pixels), especially for small, low-
  contrast, or motion-blurred features. This is a property of the measurement process, not of
  the tracker.
- **No sub-pixel ground truth.** Unlike the synthetic tests (which construct an exact known
  translation), a real video has no independently known "true" displacement - the observed
  coordinate is itself an estimate, just a human one instead of an algorithmic one.
- **Coverage is limited to the points actually validated.** A small pixel error at one or two
  points says nothing about tracking quality elsewhere in the frame or at other times in the
  video; this document and the web/script tooling deliberately avoid generalizing a per-point
  result into a global accuracy claim.
- **Single frame pair per measurement.** Each record validates one specific Frame 1 -> Frame 2
  transition; it is not a substitute for the longer-horizon trajectory/track-history
  visualization already available on the Motion Tracking page (Phase 2), which is a separate,
  non-error-quantified visualization.

## 6. Required results table (IMPLEMENTATION_PLAN.md Section 12)

Real measurements from `IMG_7272.MOV` (video_1) and `IMG_7275.MOV` (video_2). Coordinates are
`(x, y)` pixels in the convention of Section 2. Full machine-readable records:
`results/tracking/video_1/P1_record.json`, `results/tracking/video_2/P1_record.json`;
consolidated CSV: `results/metrics/phase4_tracking_validation_records.csv`.

| Video   | Point | Frame 1 -> 2 | Frame 1 coordinate | Predicted Frame 2 | Observed Frame 2 | Pixel Error |
| ------- | ----- | ------------ | ------------------- | ------------------ | ----------------- | ----------- |
| Video 1 | P1    | 275 -> 276   | (205.00, 1705.00)    | (226.35, 1705.43)   | (225.00, 1701.00)  | 4.628 px    |
| Video 2 | P1    | 1200 -> 1201 | (1127.00, 1964.00)   | (1115.61, 1950.74)  | (1110.00, 1957.00) | 8.408 px    |

**Point descriptions (what was tracked and why):**

- **Video 1, P1** - the silhouette peak/notch of the person's shirt collar against the plain
  wall background, at the moment they walk across frame roughly 9 seconds into the sample
  (frame 275 of the 30-second, 0-30 s sample). Selected via Shi-Tomasi detection restricted to
  the person's silhouette, then manually confirmed as a real, high-contrast, unambiguous edge
  feature. Frame 1 evidence: `results/tracking/video_1/P1_frame1.png`
  (zoomed: `P1_report_frame1_zoom.png`); predicted-vs-observed overlay:
  `P1_frame2_validated.png` (zoomed: `P1_report_frame2_zoom.png`).
- **Video 2, P1** - the notch apex of the letter "K" (where its diagonal arms meet its vertical
  stroke) in the "TRAILMAKER" logo text on a backpack being held up to the camera, at
  t=40.01 s in the source video (frame 1200 of 1355; frame 2 is frame 1201 at t=40.05 s). This
  falls inside video_2's declared 10.0-40.11 s optical-flow sample window (frames 300-1203,
  Section 1 of `docs/EXPERIMENTAL_RESULTS.md`). Earlier candidate points elsewhere in that same
  window (the same logo, seen at ~19.8 s and ~28.3 s) were tried first, but under dim indoor
  lighting at those points in the clip the logo graphic and printed letters were too
  motion-blurred at native 4K resolution to confidently and reproducibly identify the same
  point in both frames by eye; frame 1200 (t=40.01 s), where the lighting/focus on the backpack
  was clearer, was used instead - it is both a real frame pair from `IMG_7275.MOV` and inside
  the same 30-second sample used for the optical-flow visualization deliverable.
  Selected via Shi-Tomasi detection restricted to the logo-patch region, manually confirmed as
  a real, high-contrast printed-letter corner. Frame 1 evidence:
  `results/tracking/video_2/P1_frame1.png` (zoomed: `P1_report_frame1_zoom.png`);
  predicted-vs-observed overlay: `P1_frame2_validated.png`
  (zoomed: `P1_report_frame2_zoom.png`).

**On a second point per video:** additional candidate points were evaluated for both videos
(a second location on the person's shirt for video_1; the printed mountain-graphic edge and a
seam junction on the backpack for video_2), but each candidate that was tried was either on
ambiguous background texture or too low-contrast/motion-blurred at native 4K resolution under
indoor lighting to confidently and reproducibly identify the same point in both frames by eye.
Per IMPLEMENTATION_PLAN.md Section 12 and the assignment's "at minimum" allowance, one
defensible real point per video is reported rather than including a second, less reliable one.

**Reading the results:** both errors are small relative to the frame's 2160x3840 resolution and
each point's own predicted displacement (21.3 px for video_1 P1, 17.5 px for video_2 P1),
consistent with Lucas-Kanade tracking a high-contrast, well-localized feature well over one
frame interval (1/30 s) in each case. Video_2's error is roughly double video_1's, plausibly
because its feature (a small printed letter under dim indoor lighting, close to the camera) is
smaller and more affected by motion blur than video_1's larger-scale silhouette edge - this is
an observation about these two specific measurements, not a general claim about either video
or about Lucas-Kanade tracking overall.
