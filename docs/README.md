# Module 5-6 documentation

- `OPTICAL_FLOW_THEORY.md` - brightness constancy, the first-order Taylor expansion, the
  optical-flow constraint equation, and the aperture problem.
- `MOTION_TRACKING_DERIVATION.md` - the Lucas-Kanade overdetermined system and least-squares
  solution, the two-frame tracking problem, and how that derivation relates to the pyramidal
  OpenCV implementation used in Phase 2.
- `BILINEAR_INTERPOLATION.md` - the bilinear-interpolation derivation from two sequential 1D
  linear interpolations, plus a complete deterministic numerical worked example.
- `TRACKING_VALIDATION.md` - the professor-required two-consecutive-frame manual
  pixel-location validation: procedure, coordinate convention, point-selection method, the
  Euclidean pixel-error formula, and limitations. **PENDING USER EXPERIMENT** until the two
  real assignment videos are supplied.
- `EXPERIMENTAL_RESULTS.md` - the consolidated results write-up (video summaries, inferred
  optical-flow information, the validation table). **PENDING USER EXPERIMENT** until the two
  real assignment videos are supplied.

Later approved phases will add the structure-from-motion theory and camera-geometry notes, the
four-view SfM calculations, report notes, and the demonstration-video checklist (see
`IMPLEMENTATION_PLAN.md` for the full planned document list).

No empirical claims belong in these documents until corresponding experiments have actually
run. The bilinear-interpolation worked example and any synthetic tracking/flow examples
referenced from these documents are mathematical or software-verification examples, not
assignment experimental evidence.
