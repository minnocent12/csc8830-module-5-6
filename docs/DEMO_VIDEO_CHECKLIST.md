# Demonstration Video Checklist

A practical recording script for the Module 5-6 demonstration video required by
`MODULE_5_6_ASSIGNMENT_INSTRUCTIONS.md` Section 4 ("a screen recording, or properly captured
footage, of the working system"). This is a checklist/script to follow when recording, not a
recording performed automatically.

## Before recording

- [ ] Run `streamlit run app.py` from this repository's root and confirm it opens at
      `http://localhost:8501` with no errors in the terminal.
- [ ] Close unrelated browser tabs/notifications; use a clean browser window sized to at least
      1280x800 so the sidebar and page content are both comfortably legible.
- [ ] Have this checklist open on a second screen or printed, so the recording can follow it in
      order without narration gaps.
- [ ] Decide narration approach: live spoken narration while clicking through the app, or a
      silent screen recording with on-screen captions added afterward. Either satisfies the
      assignment; spoken narration is faster to produce.

## Recording sequence

1. **Module 5-6 web application - overview.** Show the sidebar with the Module dropdown and the
   five-page radio selector. Briefly state the two assignment questions this module covers.
2. **Optical Flow page.** With no video uploaded, show the bundled real evidence for Video 1
   and Video 2 (original frame / HSV flow / arrow overlay, magnitude statistics). Mention that
   these are the real, completed results for the two required assignment videos, loaded from
   committed results rather than requiring the large source videos to be present.
3. **Real optical-flow results (continued).** Scroll through both videos' evidence and read out
   the mean/median/max magnitude numbers, pointing out that Video 2's peak magnitude is much
   higher (closer, faster subject) while its median is much lower (mostly static background).
4. **Motion Tracking page.** With no video uploaded, show the bundled real two-consecutive-frame
   validation for both videos: Frame 1 point, predicted vs. observed Frame 2 location, and the
   real pixel error (4.628 px, 8.408 px). Explicitly state that this predicted-vs-observed
   comparison is the professor-required validation, distinct from the algorithmic
   tracking/forward-backward consistency numbers described on the page.
5. **Real pixel validation (continued).** Point out the zoomed evidence images showing the
   predicted (red) and observed (green) markers on the actual video frame.
6. **Bilinear Interpolation & Theory page.** Scroll through the brightness-constancy,
   optical-flow-constraint, aperture-problem, and Lucas-Kanade derivations. Demonstrate the
   interactive bilinear-interpolation example (the default values reproduce the verified
   I(x,y)=27.0 worked example).
7. **Structure From Motion page.** With no images uploaded, show the completed real four-view
   experiment: reference view, per-view camera/EXIF information, ORB matches and RANSAC
   inliers for each of View 2/3/4, the estimated homographies, and the boundary reconstruction
   overlay. Mention View 3's lower inlier count and that it is reported transparently, not
   hidden.
8. **Real four-view results (continued).** Show the real mathematical workout section on the
   page (source point, homography, normalized prediction, actual point, reprojection error).
9. **Experiments & Results summary.** Show this page reporting all three experiments (Video 1,
   Video 2, four-view SfM) as complete, with their key numbers consolidated in one place.
10. **GitHub repository.** Switch to a browser tab showing
    <https://github.com/minnocent12/csc8830-module-5-6> (public, accessible), and briefly show
    the repository structure (`src/`, `tests/`, `docs/`, `results/`) and the README.

## After recording

- [ ] Watch the full recording once before submitting; confirm audio (if narrated) is audible
      throughout and no page shows an error or a blank/broken screen.
- [ ] Export in a common format (MP4 recommended) and keep the file size reasonable for Google
      Classroom's upload limits, compressing if necessary without making text illegible.
- [ ] Confirm the final video, together with `docs/report/FINAL_REPORT.pdf`, are both ready for
      the Google Classroom submission described in `MODULE_5_6_ASSIGNMENT_INSTRUCTIONS.md`
      Section 5.
