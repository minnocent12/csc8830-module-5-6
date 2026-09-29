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
    foundation_page(
        "Motion Tracking",
        "Question 1: Shi-Tomasi feature detection, pyramidal Lucas-Kanade tracking between two "
        "frames, and the two-frame tracking-problem visualization will be added in a later "
        "phase.",
    )


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
