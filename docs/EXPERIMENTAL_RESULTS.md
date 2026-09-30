# Experimental Results

Consolidated report-ready results for Question 1's real-video experiments: optical-flow
evidence for each video, and the two-consecutive-frame pixel-location tracking validation.
Structure-from-motion experimental results (Question 2) are a later phase and are not covered
here.

## Status

**Optical-flow evidence and tracking validation are COMPLETE for both required videos**, using
`data/videos/video_1/IMG_7272.MOV` and `data/videos/video_2/IMG_7275.MOV`.

| Video   | Directory               | Status    |
| ------- | ------------------------ | --------- |
| Video 1 | `data/videos/video_1/`   | Processed |
| Video 2 | `data/videos/video_2/`   | Processed |

Verified with `module5_6.experiment.find_supplied_video` (also used by the Experiments &
Results web page). Every number below comes from an actual computed run against these two real
video files, recorded in `data/experiment_manifest.json`,
`results/optical_flow/video_1|video_2/*_summary.json`, and
`results/tracking/video_1|video_2/P1_record.json`. Synthetic test fixtures
(`tests/test_video.py`, `tests/test_optical_flow.py`, `tests/test_tracking.py`,
`tests/test_experiment.py`) verify the underlying code only and appear nowhere in this document.

## 1. Video summaries

| Field                 | Video 1 (`IMG_7272.MOV`)            | Video 2 (`IMG_7275.MOV`)             |
| ---------------------- | ------------------------------------ | ------------------------------------- |
| Resolution              | 2160 x 3840 (portrait)               | 2160 x 3840 (portrait)                |
| Frame rate               | 29.997 fps                           | 29.991 fps                            |
| Total frame count         | 983 (32.77 s)                        | 1355 (45.18 s)                        |
| Sample interval used      | 0.0 s - 30.0 s (frames 0-900)        | 10.0 s - 40.11 s (frames 300-1203)    |
| Sample frame pairs         | 899                                   | 902                                    |
| Decoding/codec issues       | None - decoded cleanly via OpenCV     | None - decoded cleanly via OpenCV      |
| Optical-flow video           | `results/optical_flow/video_1/video_1_optical_flow_hsv.mp4` (113.1 MB, 899 frames, verified reopenable) | `results/optical_flow/video_2/video_2_optical_flow_hsv.mp4` (58.5 MB, 902 frames, verified reopenable) |

Both videos exceed the assignment's 30-second-minimum requirement (32.77 s and 45.18 s
respectively). Each sample interval was chosen from a frame-difference motion scan (0.5 s
resolution, downscaled grayscale, mean absolute difference) covering the whole video, selecting
a clean >=30-second window containing sustained visible motion:

- **Video 1**: motion is present essentially throughout the clip (a person walks into and
  around the room), so the sample uses the first 30 seconds (0.0-30.0 s) starting from the
  beginning of the video.
- **Video 2**: the motion scan showed strong, sustained motion from roughly 5 s to 40+ s (a
  person moves close to the camera holding a backpack, with brief quieter moments around
  14.5-15.5 s, 21.5-24 s, 30.5-32.5 s, and 38.5 s). The sample uses 10.0-40.11 s (frames
  300-1203) specifically so that it contains the two-consecutive-frame tracking-validation pair
  used in Section 5 below (frames 1200-1201, t=40.01-40.05 s) inside the same declared
  optical-flow sample, per IMPLEMENTATION_PLAN.md's intent that each 30-second sample is both
  visualized as optical flow and used for the tracking validation. The scan confirms this window
  contains substantial sustained motion (repeatedly >20-55 mean absolute difference), including
  directly at the validation frame pair itself (20.33 at t=40.01 s).

Reproduce with:

```bash
python scripts/process_optical_flow.py --video data/videos/video_1/IMG_7272.MOV --video-id video_1 \
    --start-seconds 0.0 --duration-seconds 30.0
python scripts/process_optical_flow.py --video data/videos/video_2/IMG_7275.MOV --video-id video_2 \
    --start-seconds 10.0 --duration-seconds 30.1
```

Full summaries: `results/optical_flow/video_1/video_1_optical_flow_summary.json`,
`results/optical_flow/video_2/video_2_optical_flow_summary.json`.

## 2. Optical-flow magnitude statistics

| Statistic                              | Video 1  | Video 2  |
| --------------------------------------- | -------- | -------- |
| Mean magnitude (px/frame)                | 0.277    | 0.383    |
| Median magnitude (px/frame, approx.)*     | 0.186    | 0.040    |
| Max magnitude (px/frame)                   | 155.106  | 434.974  |
| Frame pairs                                  | 899      | 902      |

\* Computed across the full sample without holding every pixel's magnitude in memory
simultaneously (infeasible at 2160x3840 x ~900 frame pairs); mean and max are exact (running
sum/count and running max), while the median is the median *of each frame pair's own median* -
a documented approximation, not the exact global pixel-level median. See
`module5_6.video.render_optical_flow_video_from_path`'s docstring.

Video 2's much lower median alongside a higher mean and far higher max is consistent with what
the evidence frames show (Section 3): most of the frame is a relatively static close-up
background/fabric surface (near-zero flow for the majority of pixels, hence the low median),
while a smaller region undergoes fast, large-displacement motion (a hand/strap moving quickly
close to the camera), which pulls the mean up and produces a much larger max than Video 1's
more evenly-paced, whole-body walking motion.

## 3. Optical-flow evidence

**Video 1** (person walking across the room, frames 275-276 of the sample shown as a
representative pair):

- Original frame: `results/optical_flow/video_1/video_1_evidence_original_frame275.png`
- HSV flow visualization: `results/optical_flow/video_1/video_1_evidence_hsv_flow_275_276.png`
  - shows motion (colored, non-black pixels) concentrated on the walking person's silhouette,
    with the room's static furniture and walls correctly at or near zero (black)
- Arrow/vector overlay: `results/optical_flow/video_1/video_1_evidence_arrows_275_276.png`
- Full visualization video: `results/optical_flow/video_1/video_1_optical_flow_hsv.mp4`

**Video 2** (backpack held close to the camera, frames 1200-1201 of the sample shown as a
representative pair):

- Original frame: `results/optical_flow/video_2/video_2_evidence_original_frame1200.png`
- HSV flow visualization: `results/optical_flow/video_2/video_2_evidence_hsv_flow_1200_1201.png`
  - shows motion concentrated on the "TRAILMAKER" logo/text region and the hand/strap at the
    top of frame, with the background wall and static parts of the room near zero
- Arrow/vector overlay: `results/optical_flow/video_2/video_2_evidence_arrows_1200_1201.png`
- Full visualization video: `results/optical_flow/video_2/video_2_optical_flow_hsv.mp4`

## 4. Information inferred from optical flow

Claims below are restricted to what the generated evidence in Section 3 and the magnitude
statistics in Section 2 actually support, per IMPLEMENTATION_PLAN.md Section 6.

- **Moving vs. static regions.** In both videos, the HSV visualizations show optical flow
  correctly concentrated on the moving subject (the walking person in Video 1; the
  hand/strap/logo region in Video 2) while the static background (walls, furniture, curtains)
  shows little to no flow. This is a direct, visually verifiable observation from the linked
  evidence frames, not an inference from the magnitude numbers alone.
- **Relative motion magnitude.** Video 2's close-up handheld footage produces substantially
  larger peak displacements (max 434.97 px/frame vs. 155.11 px/frame) than Video 1's
  whole-body walking shot, consistent with the subject/camera being much closer to the lens in
  Video 2 (the same real-world motion covers more pixels when closer to the camera) and with
  faster hand/arm motion than walking pace.
- **Camera motion.** Video 2 appears to be handheld and moving with the subject (the framing
  stays close to the backpack throughout the sample), which is consistent with its much lower
  median magnitude despite the highest max - a handheld shot that tracks its subject can still
  leave most background pixels with low apparent motion between consecutive frames, while the
  subject itself (and any camera shake) produces localized spikes.
- **Regions where optical flow becomes unreliable.** The tracking-validation results in Section
  5 (an 8.4 px error for Video 2's P1 vs. a 4.6 px error for Video 1's P1) suggest Farneback/
  Lucas-Kanade flow is somewhat less reliable in Video 2's sample: the tracked feature (a small
  printed letter) is smaller in frame and more affected by motion blur under indoor lighting
  than Video 1's larger-scale silhouette edge. This is an observation about these two specific
  measurements, not a general claim about either video.

No claim is made here about acceleration, directional trends over the full 30-second sample, or
foreground/background segmentation beyond what is directly visible in the linked evidence
frames and the two point measurements in Section 5 - a full frame-by-frame directional analysis
was not performed.

## 5. Two-consecutive-frame pixel-location tracking validation

Procedure, coordinate convention, point-selection method, and the error formula are documented
in `docs/TRACKING_VALIDATION.md`. Full results table, point descriptions, and evidence figure
paths: `docs/TRACKING_VALIDATION.md` Section 6. Summary:

| Video   | Point | Frame 1 -> 2 | Timestamp     | Pixel Error |
| ------- | ----- | ------------ | ------------- | ----------- |
| Video 1 | P1    | 275 -> 276   | 9.17-9.20 s    | 4.628 px    |
| Video 2 | P1    | 1200 -> 1201 | 40.01-40.05 s  | 8.408 px    |

Both pairs fall inside their video's declared optical-flow sample: video_1's pair (9.17 s)
inside 0.0-30.0 s, and video_2's pair (40.01 s) inside 10.0-40.11 s - the video_2 sample
interval (Section 1) was specifically chosen to contain this pair. Other candidate points
elsewhere in that same video_2 window were too motion-blurred at native 4K resolution under dim
indoor lighting to confidently identify by eye; see `docs/TRACKING_VALIDATION.md` Section 6 for
why this particular moment in the window was used.

Reproduction: `docs/TRACKING_VALIDATION.md` Section 4 (script workflow) or the Motion Tracking
web page's manual-validation section. The Experiments & Results page loads these completed
records automatically from `results/tracking/`, or additional records can be uploaded there;
they can also be regenerated as CSV with `module5_6.experiment.write_records_csv`
(`results/metrics/phase4_tracking_validation_records.csv`).

## 6. Discussion

Both measured pixel errors (4.6 px and 8.4 px) are small relative to the source video's
2160x3840 resolution and to each point's own predicted displacement (21.3 px and 17.5 px
respectively) - in both cases the Lucas-Kanade prediction landed within about a quarter of the
true displacement's magnitude of the manually observed location. This is consistent with, but
does not by itself prove, generally accurate tracking: it reflects the outcome for the two
specific high-contrast points that were selected (a shirt-collar silhouette edge and a printed
letter corner), each on a single consecutive-frame pair, and should not be generalized to every
point or every frame pair in either video. Section 4's discussion of where flow appears less
reliable (Video 2's smaller, more motion-blurred feature) is a more specific and better
supported observation than any single aggregate accuracy claim would be.

## 7. Limitations

- Only one validation point per video was completed (see `docs/TRACKING_VALIDATION.md` Section
  6 for what was tried and why a second point was not included); the results above characterize
  those two specific points, not overall tracking accuracy for either video.
- The manually observed coordinates carry the precision limits of visual pixel-grid inspection
  on blurry, low-light, close-up footage (Video 2 especially); see
  `docs/TRACKING_VALIDATION.md` Section 5.
- The median-magnitude statistic in Section 2 is an approximation at this data scale (see the
  note there); it is not used for any pass/fail judgment, only as descriptive context.
- No structure-from-motion, camera-parameter, or four-view result appears in this document;
  those remain a later phase.

## 8. Reproducibility notes

- `data/experiment_manifest.json` records per-video status (`processed` for both videos as of
  this experiment), path, resolution/fps, the sample interval used, and the optical-flow output
  path, plus the two completed `TrackingValidationRecord` entries; its schema is
  `module5_6.experiment.default_experiment_manifest`.
- All frame indices, coordinates, and errors above are traceable to a specific video file, frame
  pair, and (for the observed coordinate) an `observation_method` string recorded on the
  corresponding `TrackingValidationRecord` JSON file under `results/tracking/`.
- Automated tests (`tests/test_video.py`, `tests/test_optical_flow.py`, `tests/test_tracking.py`,
  `tests/test_experiment.py`) verify the underlying code against synthetic fixtures with known
  answers; they are software-correctness evidence, not a substitute for the real-video results
  in this document, and none of their values appear above.
