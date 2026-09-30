# Module 5-6 data

- `videos/video_1/IMG_7272.MOV` and `videos/video_2/IMG_7275.MOV` are the two real assignment
  videos for Question 1 (2160x3840, ~30 fps, 32.77 s and 45.18 s respectively - both exceed the
  30-second-minimum requirement). They are gitignored (`data/videos/video_1/*` /
  `video_2/*` in `.gitignore`) and present only in this local working copy; they are not
  committed to the repository. See `docs/EXPERIMENTAL_RESULTS.md` for the sample intervals used
  and full results, and "Preserve raw videos" in the root task instructions for why they stay
  local-only and untouched.
- `sfm/view_1/` through `sfm/view_4/` will hold the four-viewpoint images of the chosen flat/2D
  planar object, per Question 2 - still pending for a later phase.
- Synthetic arrays used by automated tests are fixtures, not experimental evidence.

Until the four-view SfM images are supplied, structure-from-motion results remain
**PENDING USER EXPERIMENT**. Optical-flow and tracking-validation results for Question 1 are
complete for both videos - see `docs/EXPERIMENTAL_RESULTS.md` and `docs/TRACKING_VALIDATION.md`.

`experiment_manifest.json` records per-video Phase 4 experiment status (`path`, `status`,
resolution/fps, sample interval, optical-flow output path) and the completed validation
records; its schema is `module5_6.experiment.default_experiment_manifest`. Both required videos
now show `status: "processed"` with real metadata and one completed `TrackingValidationRecord`
each.
