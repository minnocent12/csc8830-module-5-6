# Module 5-6 data

- `videos/video_1/IMG_7272.MOV` and `videos/video_2/IMG_7275.MOV` are the two real assignment
  videos for Question 1 (2160x3840, ~30 fps, 32.77 s and 45.18 s respectively - both exceed the
  30-second-minimum requirement). They are gitignored (`data/videos/video_1/*` /
  `video_2/*` in `.gitignore`) and present only in this local working copy; they are not
  committed to the repository. See `docs/EXPERIMENTAL_RESULTS.md` for the sample intervals used
  and full results, and "Preserve raw videos" in the root task instructions for why they stay
  local-only and untouched.
- `sfm/view_1/IMG_7283.JPG` through `sfm/view_4/IMG_7286.JPG` are the four real viewpoint
  images of the chosen flat/2D planar object (a paperback book's front cover) for Question 2,
  captured with an iPhone 15 Pro Max. They are gitignored (`data/sfm/view_N/*` in
  `.gitignore`) and present only in this local working copy - the derived real results and
  report figures under `results/sfm/` are committed instead, so the SfM web page and this
  documentation do not depend on the original photos being present. The registration software
  (`module5_6.features`, `homography`, `camera`, `geometry`, `sfm`) was built and tested
  against synthetic fixtures in Phase 5, then run on these real images in Phase 6; see
  `docs/STRUCTURE_FROM_MOTION_THEORY.md` and `docs/EXPERIMENTAL_RESULTS.md` Section 9. Each
  view's real camera information (device, focal length from EXIF, user-recorded approximate
  position/orientation/distance) is recorded with `module5_6.camera.ViewMetadata` and in
  `results/sfm/sfm_summary.json` - every physical field the real data does not support stays
  `None`/omitted, never inferred from image dimensions.
- Synthetic arrays used by automated tests are fixtures, not experimental evidence.

Optical-flow and tracking-validation results for Question 1, and the four-view SfM experiment
for Question 2, are **complete** - see `docs/EXPERIMENTAL_RESULTS.md`, `docs/TRACKING_VALIDATION.md`,
and `docs/SFM_CALCULATIONS.md` Sections 6-7.

`experiment_manifest.json` records per-video Phase 4 experiment status (`path`, `status`,
resolution/fps, sample interval, optical-flow output path) and the completed validation
records (both required videos show `status: "processed"` with real metadata and one completed
`TrackingValidationRecord` each), plus the real Phase 6 `sfm` section (reference view, per-view
camera/boundary data, per-view registration/homography/reprojection results, and the boundary
reconstruction) produced by `scripts/process_sfm_experiment.py`.
