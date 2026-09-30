# Module 5-6 results

Optical-flow and tracking-validation experiments have been run against both real assignment
videos (`data/videos/video_1/IMG_7272.MOV`, `data/videos/video_2/IMG_7275.MOV`) - see
`docs/EXPERIMENTAL_RESULTS.md` for the full write-up.

- `optical_flow/<video_id>/` (written by `scripts/process_optical_flow.py`) contains
  `<video_id>_optical_flow_hsv.mp4` (the required 30-second flow-visualization video),
  `<video_id>_optical_flow_summary.json` (sample metadata and magnitude statistics), and
  `<video_id>_evidence_*.png` (representative original/HSV-flow/arrow-overlay report figures).
- `tracking/<video_id>/` (written by `scripts/validate_tracking.py` or the Motion Tracking
  page's manual-validation section) contains `<point_label>_record.json` (the validation
  record), `<point_label>_frame1.png`, `<point_label>_frame2_predicted.png`,
  `<point_label>_frame2_validated.png`, and zoomed `<point_label>_report_frame1|2_zoom.png`
  report figures.
- `metrics/phase4_tracking_validation_records.csv` is the consolidated CSV of all completed
  validation records (`module5_6.experiment.write_records_csv`).

The small evidence figures (`*.png`, downscaled to a ~900px max dimension), validation records
(`*.json`), and the consolidated CSV are committed - they are real, derived report evidence,
matching Module 3/4's convention of tracking only small, intentionally-selected evidence. The
large generated optical-flow videos (`*.mp4`, ~50-110 MB each) stay gitignored and local-only,
same as the raw source videos, and are reproducible from them with
`scripts/process_optical_flow.py`.

`sfm/` remains empty: the Phase 5 planar-registration software
(`module5_6.features`/`homography`/`camera`/`geometry`/`sfm`) is implemented and tested
against synthetic fixtures only (see `docs/STRUCTURE_FROM_MOTION_THEORY.md`,
`docs/SFM_CALCULATIONS.md`), and writes no output here until it is run against the real
four-view images (`data/sfm/view_1/` through `view_4/`, still pending). Missing measurements
are marked PENDING USER EXPERIMENT rather than fabricated.
