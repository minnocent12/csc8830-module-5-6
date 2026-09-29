"""Module 5-6 Streamlit pages and the get_pages provider.

Phase 1 implements the Optical Flow page (Question 1: computing and visualizing dense
Farneback optical flow on a sampled video interval). Motion Tracking, Bilinear Interpolation
& Theory, Structure From Motion, and Experiments & Results remain pending-safe placeholders
for later approved phases (see IMPLEMENTATION_PLAN.md).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from module5_6.io_utils import to_grayscale
from module5_6.optical_flow import (
    compute_farneback_flow,
    draw_flow_arrows,
    flow_magnitude_angle,
    flow_to_hsv_bgr,
    summarize_flow_magnitudes,
)
from module5_6.tracking import (
    LucasKanadeParams,
    ShiTomasiParams,
    detect_features,
    displacement_magnitude,
    draw_displacement_vectors,
    draw_tracked_points,
    draw_trajectories,
    forward_backward_validate,
    track_points,
    track_trajectories,
    valid_forward_backward_mask,
)
from module5_6.video import (
    MINIMUM_SAMPLE_DURATION_SECONDS,
    compute_sample_frame_range,
    get_video_metadata,
    read_frame_range,
    render_optical_flow_video,
)
from module5_6.webapp._page import PageSpec
from module5_6.webapp.ui import VIDEO_TYPES, foundation_page, pending_experiment_banner

_MODULE = "Module 5-6"


def _bgr_to_rgb(image):
    """Reverse the channel order for display with st.image; never mutates the input."""
    return image[:, :, ::-1]


def _optical_flow_page() -> None:
    st.header("Optical Flow")
    st.info(
        "Question 1: upload a video to compute dense Farneback optical flow over a sampled "
        "interval and visualize it as a video. Sparse Lucas-Kanade point tracking is on the "
        "Motion Tracking page (a later phase)."
    )
    upload = st.file_uploader("Video", type=VIDEO_TYPES)
    if upload is None:
        pending_experiment_banner(
            "Upload a video to compute optical flow. The two videos required by the "
            "assignment (each with a 30-second sample containing motion) are PENDING USER "
            "EXPERIMENT until supplied."
        )
        return

    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_input:
        tmp_input.write(upload.getvalue())
        input_path = Path(tmp_input.name)

    try:
        try:
            metadata = get_video_metadata(input_path)
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Could not read the uploaded video: {exc}")
            return

        st.caption(
            f"Input: {upload.name} | {metadata.width} x {metadata.height} pixels | "
            f"{metadata.fps:.2f} fps | {metadata.frame_count} frames | "
            f"{metadata.duration_seconds:.2f} s"
        )

        st.subheader("Sample interval")
        max_start = max(0.0, metadata.duration_seconds - 1.0)
        start_seconds = float(
            st.number_input(
                "Start time (seconds)", min_value=0.0, max_value=max_start, value=0.0, step=1.0
            )
        )
        enforce_minimum = st.checkbox(
            "Require the assignment minimum of 30 seconds",
            value=True,
            help="Question 1 requires at least a 30-second sample from each video.",
        )
        max_duration = max(1.0, metadata.duration_seconds - start_seconds)
        default_duration = min(MINIMUM_SAMPLE_DURATION_SECONDS, max_duration)
        duration_seconds = float(
            st.number_input(
                "Sample duration (seconds)",
                min_value=1.0,
                max_value=max_duration,
                value=default_duration,
                step=1.0,
            )
        )

        st.subheader("Visualization settings")
        mode_label = st.radio("Visualization mode", ["HSV color", "Arrow overlay"], horizontal=True)
        mode = "hsv" if mode_label == "HSV color" else "arrows"
        vector_spacing = int(
            st.slider("Vector spacing (pixels)", min_value=4, max_value=48, value=16, step=2)
        )
        magnitude_threshold = float(
            st.slider(
                "Arrow magnitude threshold (pixels)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
            )
        )
        max_preview_frames = int(
            st.slider(
                "Max frames to process (interactive-preview limit)",
                min_value=10,
                max_value=300,
                value=90,
                step=10,
                help=(
                    "Bounds compute time for this live demo. A later phase's scripts will "
                    "process the full required sample for the actual assignment submission."
                ),
            )
        )

        if not st.button("Compute optical flow", type="primary"):
            pending_experiment_banner(
                "Set the sample interval and visualization settings, then run the pipeline to "
                "view results."
            )
            return

        try:
            minimum = MINIMUM_SAMPLE_DURATION_SECONDS if enforce_minimum else None
            start_frame, end_frame = compute_sample_frame_range(
                metadata,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                minimum_duration_seconds=minimum,
            )
        except ValueError as exc:
            st.error(f"Invalid sample interval: {exc}")
            return

        capped = (end_frame - start_frame) > max_preview_frames
        if capped:
            end_frame = start_frame + max_preview_frames

        try:
            frames = read_frame_range(input_path, start_frame, end_frame)
        except ValueError as exc:
            st.error(f"Could not read the requested frames: {exc}")
            return

        if len(frames) < 2:
            st.error(
                "At least two frames are required to compute optical flow; widen the sample "
                "interval."
            )
            return

        if capped:
            st.caption(
                f"Interactive preview capped to the first {len(frames)} frames "
                f"(~{len(frames) / metadata.fps:.1f} s) of the requested "
                f"{duration_seconds:.1f} s sample. This cap affects only this live demo, not "
                "a later phase's full-sample processing."
            )
        if not enforce_minimum and duration_seconds < MINIMUM_SAMPLE_DURATION_SECONDS:
            st.warning(
                "This sample is shorter than the assignment's required 30 seconds. It is shown "
                "for interactive preview only and does not by itself satisfy Question 1."
            )

        grays = [to_grayscale(frame) for frame in frames]
        flows = [compute_farneback_flow(grays[i], grays[i + 1]) for i in range(len(grays) - 1)]
        magnitudes = [flow_magnitude_angle(flow)[0] for flow in flows]
        stats = summarize_flow_magnitudes(magnitudes)

        st.subheader("Original sample frames")
        c1, c2 = st.columns(2)
        with c1:
            st.image(
                _bgr_to_rgb(frames[0]),
                caption=f"Frame {start_frame} (sample start)",
                width="stretch",
            )
        with c2:
            st.image(
                _bgr_to_rgb(frames[-1]),
                caption=f"Frame {start_frame + len(frames) - 1} (sample end)",
                width="stretch",
            )

        mid = len(flows) // 2
        st.subheader("Optical-flow visualization (representative frame pair)")
        c1, c2 = st.columns(2)
        with c1:
            st.image(
                _bgr_to_rgb(flow_to_hsv_bgr(flows[mid])),
                caption="HSV color: hue = direction, value = magnitude",
                width="stretch",
            )
        with c2:
            st.image(
                _bgr_to_rgb(
                    draw_flow_arrows(
                        frames[mid],
                        flows[mid],
                        step=vector_spacing,
                        min_magnitude=magnitude_threshold,
                    )
                ),
                caption="Sampled vector/arrow overlay",
                width="stretch",
            )

        st.subheader("Optical-flow video export")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "optical_flow.mp4"
            video_bytes = None
            try:
                render_optical_flow_video(
                    frames,
                    output_path,
                    fps=metadata.fps,
                    mode=mode,
                    arrow_step=vector_spacing,
                    arrow_min_magnitude=magnitude_threshold,
                )
                video_bytes = output_path.read_bytes()
            except ValueError as exc:
                st.error(f"Could not render the optical-flow video: {exc}")
            if video_bytes:
                st.video(video_bytes)
                st.download_button(
                    "Download optical-flow video",
                    data=video_bytes,
                    file_name="module5_6_optical_flow.mp4",
                    mime="video/mp4",
                )

        st.subheader("Selected statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frame pairs", stats.frame_pairs)
        c2.metric("Mean |flow| (px)", f"{stats.mean_magnitude:.3f}")
        c3.metric("Median |flow| (px)", f"{stats.median_magnitude:.3f}")
        c4.metric("Max |flow| (px)", f"{stats.max_magnitude:.3f}")
        st.caption(
            "These statistics are computed live from whatever video was uploaded above; they "
            "are exploratory tooling, not the assignment's required two-frame pixel tracking "
            "validation (a later approved phase). The two required assignment videos remain "
            "PENDING USER EXPERIMENT until supplied."
        )
    finally:
        input_path.unlink(missing_ok=True)


def _motion_tracking_page() -> None:
    st.header("Motion Tracking")
    st.info(
        "Question 1: Shi-Tomasi feature detection plus pyramidal Lucas-Kanade tracking between "
        "consecutive frames, following the two-frame tracking problem from "
        "IMPLEMENTATION_PLAN.md Section 10 (find P' = (x+u, y+v) in Frame 2 for each point "
        "P = (x, y) in Frame 1). The formal brightness-constancy/Lucas-Kanade derivation and "
        "manual pixel-location validation are later phases."
    )
    upload = st.file_uploader("Video", type=VIDEO_TYPES, key="tracking_upload")
    if upload is None:
        pending_experiment_banner(
            "Upload a video to detect and track features. The two videos required by the "
            "assignment, and their required two-consecutive-frame pixel validation, are "
            "PENDING USER EXPERIMENT until supplied."
        )
        return

    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_input:
        tmp_input.write(upload.getvalue())
        input_path = Path(tmp_input.name)

    try:
        try:
            metadata = get_video_metadata(input_path)
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Could not read the uploaded video: {exc}")
            return

        st.caption(
            f"Input: {upload.name} | {metadata.width} x {metadata.height} pixels | "
            f"{metadata.fps:.2f} fps | {metadata.frame_count} frames | "
            f"{metadata.duration_seconds:.2f} s"
        )

        st.subheader("Frame selection")
        max_start = max(0.0, metadata.duration_seconds - (2.0 / metadata.fps))
        start_seconds = float(
            st.number_input(
                "Start time (seconds)",
                min_value=0.0,
                max_value=max_start,
                value=0.0,
                step=1.0,
                key="tracking_start_seconds",
            )
        )
        frames_to_load = int(
            st.slider(
                "Frames to load (track-history length)",
                min_value=2,
                max_value=60,
                value=15,
                help=(
                    "Frame 1 and Frame 2 of the two-frame tracking demo are always the first "
                    "two consecutive frames of this loaded window; the full window is used for "
                    "the track-history/trajectory visualization."
                ),
                key="tracking_frames_to_load",
            )
        )

        with st.expander("Shi-Tomasi feature-detection settings", expanded=False):
            max_corners = int(st.slider("Max corners", 10, 300, 100, key="tracking_max_corners"))
            quality_level = float(
                st.slider("Quality level", 0.01, 0.5, 0.3, step=0.01, key="tracking_quality_level")
            )
            min_distance = float(
                st.slider("Min distance between corners (pixels)", 1.0, 30.0, 7.0, key="tracking_min_distance")
            )
            block_size = int(
                st.select_slider("Block size", options=[3, 5, 7, 9, 11], value=7, key="tracking_block_size")
            )
            use_harris = st.checkbox("Use Harris corner detector", value=False, key="tracking_use_harris")

        with st.expander("Lucas-Kanade tracking settings", expanded=False):
            win_size = int(
                st.select_slider("Window size", options=[15, 21, 31, 41], value=21, key="tracking_win_size")
            )
            max_level = int(st.slider("Pyramid levels", 0, 5, 3, key="tracking_max_level"))
            max_iterations = int(
                st.slider("Max iterations", 5, 100, 30, key="tracking_max_iterations")
            )
            epsilon = float(
                st.slider("Convergence epsilon", 0.001, 0.1, 0.01, step=0.001, key="tracking_epsilon")
            )
            validate_fb = st.checkbox(
                "Validate tracks with forward-backward error",
                value=True,
                help="Tracks points forward then backward; a large drift flags an unreliable track.",
                key="tracking_validate_fb",
            )
            max_fb_error = float(
                st.slider(
                    "Max forward-backward error (pixels)",
                    0.1,
                    5.0,
                    1.0,
                    step=0.1,
                    key="tracking_max_fb_error",
                )
            )

        shi_tomasi_params = ShiTomasiParams(
            max_corners=max_corners,
            quality_level=quality_level,
            min_distance=min_distance,
            block_size=block_size,
            use_harris_detector=use_harris,
        )
        lk_params = LucasKanadeParams(
            win_size=(win_size, win_size), max_level=max_level, max_iterations=max_iterations, epsilon=epsilon
        )

        if not st.button("Detect and track features", type="primary"):
            pending_experiment_banner(
                "Set the frame window and detector/tracker settings, then run to view results."
            )
            return

        start_frame = int(round(start_seconds * metadata.fps))
        start_frame = max(0, min(start_frame, max(0, metadata.frame_count - 2)))
        end_frame = min(metadata.frame_count, start_frame + frames_to_load)

        try:
            frames = read_frame_range(input_path, start_frame, end_frame)
        except ValueError as exc:
            st.error(f"Could not read the requested frames: {exc}")
            return

        if len(frames) < 2:
            st.error("At least two frames are required to track features; choose an earlier start time.")
            return

        grays = [to_grayscale(frame) for frame in frames]
        features = detect_features(grays[0], params=shi_tomasi_params)
        if features.shape[0] == 0:
            st.warning(
                "No Shi-Tomasi features were detected on Frame 1. Try a lower quality level or "
                "a smaller min-distance."
            )
            return

        # Two-frame tracking problem: Frame 1 (grays[0]) -> Frame 2 (grays[1]).
        tracked = track_points(grays[0], grays[1], features, params=lk_params)

        if validate_fb:
            fb_result = forward_backward_validate(grays[0], grays[1], features, params=lk_params)
            mask = valid_forward_backward_mask(fb_result, max_fb_error=max_fb_error)
        else:
            fb_result = None
            mask = tracked.status

        valid_previous = tracked.previous_points[mask]
        valid_next = tracked.next_points[mask]
        valid_error = tracked.error[mask]
        valid_fb_error = fb_result.fb_error[mask] if fb_result is not None else None

        st.subheader("Frame 1 and Frame 2")
        c1, c2 = st.columns(2)
        with c1:
            st.image(
                _bgr_to_rgb(draw_tracked_points(frames[0], features, color=(0, 255, 0))),
                caption=f"Frame {start_frame}: detected Shi-Tomasi features ({features.shape[0]})",
                width="stretch",
            )
        with c2:
            st.image(
                _bgr_to_rgb(draw_tracked_points(frames[1], valid_next, color=(0, 0, 255))),
                caption=f"Frame {start_frame + 1}: tracked features ({valid_next.shape[0]} valid)",
                width="stretch",
            )

        if valid_previous.shape[0] == 0:
            st.warning(
                "No tracks passed validation. Try relaxing the forward-backward threshold or "
                "adjusting the Lucas-Kanade window size."
            )
        else:
            st.subheader("Displacement vectors (Frame 1 -> Frame 2)")
            st.image(
                _bgr_to_rgb(draw_displacement_vectors(frames[0], valid_previous, valid_next)),
                caption="Displacement vector (u, v) for each valid tracked point",
                width="stretch",
            )

            displacements = valid_next - valid_previous
            magnitudes = displacement_magnitude(displacements)

            st.subheader("Selected statistics")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Detected features", features.shape[0])
            s2.metric("Valid tracks", valid_previous.shape[0])
            s3.metric("Mean |displacement| (px)", f"{float(magnitudes.mean()):.3f}")
            s4.metric("Max |displacement| (px)", f"{float(magnitudes.max()):.3f}")

            st.subheader("Point coordinates and tracking-quality metrics")
            st.caption(
                "'LK consistency error' and 'Forward-backward consistency (px)' below are "
                "algorithmic self-consistency signals reported by this run's Lucas-Kanade "
                "tracker - they measure how well the tracker's own model fit, and how well a "
                "point tracked forward then backward returns to where it started. They are "
                "**not** the assignment's required pixel-location validation, which compares a "
                "predicted location against an actual observed location manually identified in "
                "a real video frame. That comparison is a later phase and remains "
                "PENDING USER EXPERIMENT for the two required assignment videos."
            )
            max_rows = 200
            rows = []
            for index in range(min(valid_previous.shape[0], max_rows)):
                row = {
                    "Point": index,
                    "Frame1 x": round(float(valid_previous[index, 0]), 2),
                    "Frame1 y": round(float(valid_previous[index, 1]), 2),
                    "Frame2 x (predicted)": round(float(valid_next[index, 0]), 2),
                    "Frame2 y (predicted)": round(float(valid_next[index, 1]), 2),
                    "u": round(float(displacements[index, 0]), 2),
                    "v": round(float(displacements[index, 1]), 2),
                    "Magnitude (px)": round(float(magnitudes[index]), 2),
                    "LK consistency error (algorithmic)": round(float(valid_error[index]), 4),
                }
                if valid_fb_error is not None:
                    row["Forward-backward consistency (px)"] = round(float(valid_fb_error[index]), 3)
                rows.append(row)
            st.dataframe(rows, width="stretch")
            if valid_previous.shape[0] > max_rows:
                st.caption(f"Showing the first {max_rows} of {valid_previous.shape[0]} valid tracks.")
            st.caption(
                "Frame 2 coordinates above are the tracker's *predicted* location, not a "
                "manually observed one; no actual observed pixel location has been recorded "
                "for these points."
            )

        st.subheader("Track history across the loaded frames")
        trajectories = track_trajectories(grays, shi_tomasi_params=shi_tomasi_params, lk_params=lk_params)
        alive_full_length = sum(1 for t in trajectories if len(t.positions) == len(grays))
        st.image(
            _bgr_to_rgb(draw_trajectories(frames[-1], trajectories)),
            caption=(
                f"Trajectories over {len(frames)} frames (frames {start_frame} - "
                f"{start_frame + len(frames) - 1}); {alive_full_length} of {len(trajectories)} "
                "tracked points survived the full window"
            ),
            width="stretch",
        )
        st.caption(
            "These trajectories are computed live from whatever video was uploaded above; they "
            "are exploratory tooling, not the assignment's required experimental evidence. The "
            "two required assignment videos and their pixel-level tracking validation remain "
            "PENDING USER EXPERIMENT until supplied."
        )
    finally:
        input_path.unlink(missing_ok=True)


def _theory_page() -> None:
    foundation_page(
        "Bilinear Interpolation & Theory",
        "Question 1: the brightness-constancy derivation, the optical-flow constraint "
        "equation, the aperture problem, the Lucas-Kanade least-squares derivation, and an "
        "interactive bilinear-interpolation demonstration will be added in a later phase.",
    )


def _sfm_page() -> None:
    foundation_page(
        "Structure From Motion",
        "Question 2: four-viewpoint image upload, feature correspondence, planar homography "
        "estimation, reprojection validation, and recovered-boundary visualization will be "
        "added in a later phase.",
    )


def _experiments_page() -> None:
    foundation_page(
        "Experiments & Results",
        "Consolidated video summaries, tracking-error tables, four-view SfM experiment "
        "results, camera information, and reprojection results will be added once the real "
        "videos and four-view images have been processed in later phases.",
    )


def get_pages() -> list[PageSpec]:
    """Return the Module 5-6 pages for standalone or shared-dashboard hosts."""
    return [
        PageSpec(_MODULE, "Optical Flow", 10, _optical_flow_page),
        PageSpec(_MODULE, "Motion Tracking", 20, _motion_tracking_page),
        PageSpec(_MODULE, "Bilinear Interpolation & Theory", 30, _theory_page),
        PageSpec(_MODULE, "Structure From Motion", 40, _sfm_page),
        PageSpec(_MODULE, "Experiments & Results", 50, _experiments_page),
    ]
