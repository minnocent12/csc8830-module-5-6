# Module 5-6 data

The video and structure-from-motion directories are intentionally empty in Phase 0.

- `videos/video_1/` and `videos/video_2/` will hold the two user-captured videos (or a
  pointer/manifest to them if the raw files are too large to commit), each with at least a
  30-second sample containing motion, per Question 1.
- `sfm/view_1/` through `sfm/view_4/` will hold the four-viewpoint images of the chosen flat/2D
  planar object, per Question 2.
- User videos and images are not committed by default. A bundled sample may be added only
  after its source and the user's choice to commit it are confirmed, per the root
  `AGENTS.md` -> "Bundled real-sample fallback".
- Synthetic arrays used by automated tests are fixtures, not experimental evidence.

Until real videos and four-view images are supplied, all optical-flow, tracking, and
structure-from-motion results remain **PENDING USER EXPERIMENT**.
