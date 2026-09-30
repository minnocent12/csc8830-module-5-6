# Module 5-6 results

No optical-flow or tracking-validation experiment has been run against a real assignment video
yet - both `data/videos/video_1/` and `data/videos/video_2/` are still empty placeholders, so
these directories remain empty (only `.gitkeep` files are tracked).

Once a real video is supplied, `scripts/process_optical_flow.py` writes
`optical_flow/<video_id>/<video_id>_optical_flow_<mode>.mp4` and
`<video_id>_optical_flow_summary.json`, and `scripts/validate_tracking.py` (or the Motion
Tracking page's manual-validation section) writes `tracking/<video_id>/<point_label>_frame1.png`,
`<point_label>_frame2_predicted.png`, `<point_label>_frame2_validated.png`, and
`<point_label>_record.json`. Structure-from-motion outputs (`sfm/`) remain a later phase.
Missing measurements are marked PENDING USER EXPERIMENT rather than fabricated.
