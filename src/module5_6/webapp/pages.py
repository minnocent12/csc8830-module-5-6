"""Module 5-6 Streamlit pages and the get_pages provider.

Phase 1 implements the Optical Flow page (Question 1: computing and visualizing dense
Farneback optical flow on a sampled video interval). Motion Tracking, Bilinear Interpolation
& Theory, Structure From Motion, and Experiments & Results remain pending-safe placeholders
for later approved phases (see IMPLEMENTATION_PLAN.md).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from module5_6.experiment import (
    TrackingValidationRecord,
    default_experiment_manifest,
    draw_validation_overlay,
    find_supplied_video,
    record_observation,
    records_to_table,
)
from module5_6.interpolation import bilinear_interpolate_corners, bilinear_weights
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
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _bgr_to_rgb(image):
    """Reverse the channel order for display with st.image; never mutates the input."""
    return image[:, :, ::-1]


def _bilinear_diagram(i00: float, i10: float, i01: float, i11: float, alpha: float, beta: float):
    """A schematic (not real-image) diagram of the four corners and the query point."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    corners = {
        (0.0, 0.0): ("I00", i00),
        (1.0, 0.0): ("I10", i10),
        (0.0, 1.0): ("I01", i01),
        (1.0, 1.0): ("I11", i11),
    }
    for (cx, cy), (label, value) in corners.items():
        ax.scatter([cx], [cy], s=140, color="tab:blue", zorder=3)
        offset = (10, 10) if cy == 0.0 else (10, -16)
        ax.annotate(f"{label} = {value:g}", (cx, cy), textcoords="offset points", xytext=offset)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="gray", linewidth=1)
    ax.scatter([alpha], [beta], s=160, color="tab:red", marker="x", zorder=4)
    ax.annotate(
        f"(x, y): alpha={alpha:.2f}, beta={beta:.2f}",
        (alpha, beta),
        textcoords="offset points",
        xytext=(10, -4),
        color="tab:red",
    )
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(1.3, -0.3)  # inverted: y increases downward, matching image row convention
    ax.set_xlabel("x  (alpha = x - x0)")
    ax.set_ylabel("y  (beta = y - y0)")
    ax.set_title("Four neighboring pixels and the query point (schematic)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


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

            st.subheader("Two-frame pixel-location validation (manual, professor-required)")
            st.caption(
                "IMPLEMENTATION_PLAN.md Section 12: pick a point above, see its algorithmic "
                "Lucas-Kanade prediction, then **you** visually determine and enter where that "
                "point actually is in Frame 2. Unlike 'LK consistency error' and "
                "'Forward-backward consistency' above - which never look at Frame 2 as an "
                "image - this produces a meaningful pixel error only if the observed "
                "coordinate truly comes from inspecting Frame 2, not from accepting a default "
                "or copying the prediction."
            )
            point_index = int(
                st.number_input(
                    "Point index to validate (row number from the table above)",
                    min_value=0,
                    max_value=valid_previous.shape[0] - 1,
                    value=0,
                    step=1,
                    key="tracking_validation_point_index",
                )
            )
            v1, v2 = st.columns(2)
            video_id = v1.text_input("Video ID for this record", value="video_1", key="tracking_validation_video_id")
            point_label = v2.text_input("Point label", value="P1", key="tracking_validation_point_label")
            x1, y1 = float(valid_previous[point_index, 0]), float(valid_previous[point_index, 1])
            predicted_x2, predicted_y2 = float(valid_next[point_index, 0]), float(valid_next[point_index, 1])
            st.write(
                f"Frame 1 point: ({x1:.2f}, {y1:.2f})  |  Algorithmic predicted Frame 2 point: "
                f"({predicted_x2:.2f}, {predicted_y2:.2f})"
            )

            crop_half = 60
            crop_center_x, crop_center_y = int(round(predicted_x2)), int(round(predicted_y2))
            frame2_height, frame2_width = frames[1].shape[:2]
            x_lo = max(0, crop_center_x - crop_half)
            x_hi = min(frame2_width, crop_center_x + crop_half)
            y_lo = max(0, crop_center_y - crop_half)
            y_hi = min(frame2_height, crop_center_y + crop_half)
            crop = frames[1][y_lo:y_hi, x_lo:x_hi]
            st.image(
                _bgr_to_rgb(crop),
                caption=(
                    f"Frame 2 crop around the prediction (x in [{x_lo}, {x_hi}), "
                    f"y in [{y_lo}, {y_hi})) - inspect this to determine the actual location"
                ),
                width="stretch",
            )

            confirmed = st.checkbox(
                "I have visually inspected Frame 2 above and the coordinates below reflect "
                "what I actually observed (not copied from the prediction)",
                value=False,
                key="tracking_validation_confirmed",
            )
            oc1, oc2 = st.columns(2)
            observed_x = float(
                oc1.number_input("Observed Frame 2 x", value=x1, step=1.0, key="tracking_validation_observed_x")
            )
            observed_y = float(
                oc2.number_input("Observed Frame 2 y", value=y1, step=1.0, key="tracking_validation_observed_y")
            )

            if confirmed:
                validation_record = record_observation(
                    TrackingValidationRecord(
                        video_id=video_id,
                        point_label=point_label,
                        frame1_index=start_frame,
                        frame2_index=start_frame + 1,
                        x1=x1,
                        y1=y1,
                        predicted_x2=predicted_x2,
                        predicted_y2=predicted_y2,
                    ),
                    observed_x=observed_x,
                    observed_y=observed_y,
                    method="Streamlit Motion Tracking page - manual visual inspection",
                )
                st.metric("Pixel error e (predicted vs. observed)", f"{validation_record.pixel_error:.3f} px")
                st.image(
                    _bgr_to_rgb(draw_validation_overlay(frames[1], validation_record)),
                    caption="Predicted (red) vs. observed (green) Frame 2 location",
                    width="stretch",
                )
                st.download_button(
                    "Download validation record (JSON)",
                    data=json.dumps(validation_record.to_dict(), indent=2),
                    file_name=f"{video_id}_{point_label}_validation_record.json",
                    mime="application/json",
                    key="tracking_validation_download",
                )
                st.caption(
                    "Save this record under results/tracking/<video_id>/, or upload it on the "
                    "Experiments & Results page, to include it in the consolidated validation "
                    "table. This pixel error is real and computed from the coordinates you "
                    "entered - it is only meaningful evidence if the observed coordinate truly "
                    "came from inspecting Frame 2."
                )
            else:
                pending_experiment_banner(
                    "Manual pixel-location validation for this point is PENDING USER "
                    "EXPERIMENT until you inspect Frame 2 above, enter the actual observed "
                    "coordinate, and check the confirmation box."
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
    st.header("Bilinear Interpolation & Theory")
    st.info(
        "Question 1's required theory: brightness constancy through the optical-flow "
        "constraint equation and the aperture problem, the Lucas-Kanade least-squares "
        "tracking derivation and its relationship to the Phase 2 OpenCV implementation, and "
        "the bilinear-interpolation derivation with an interactive demonstration. Full "
        "write-ups: docs/OPTICAL_FLOW_THEORY.md, docs/MOTION_TRACKING_DERIVATION.md, "
        "docs/BILINEAR_INTERPOLATION.md."
    )

    st.subheader("1. Brightness constancy")
    st.markdown(
        "A point's brightness is assumed not to change as it moves between frames - only its "
        "position changes. If a point at `(x, y)` at time `t` moves by `(Δx, Δy)` "
        "over a small interval `Δt`:"
    )
    st.latex(r"I(x, y, t) = I(x + \Delta x,\; y + \Delta y,\; t + \Delta t)")

    st.subheader("2. First-order Taylor expansion and the optical-flow constraint")
    st.markdown("Expanding the right-hand side to first order:")
    st.latex(
        r"I(x+\Delta x,\, y+\Delta y,\, t+\Delta t) \approx "
        r"I(x,y,t) + I_x \Delta x + I_y \Delta y + I_t \Delta t"
    )
    st.markdown(
        "Subtracting `I(x, y, t)` from both sides, dividing by `Δt`, and defining "
        "`u = Δx / Δt`, `v = Δy / Δt` gives the **optical-flow constraint "
        "equation**:"
    )
    st.latex(r"I_x u + I_y v + I_t = 0")
    st.caption("Full step-by-step derivation: docs/OPTICAL_FLOW_THEORY.md, Sections 2-4.")

    st.subheader("3. The aperture problem")
    st.markdown(
        "One pixel gives one equation, `Ix*u + Iy*v = -It`, in two unknowns `(u, v)`: only "
        "the motion component along the local gradient direction is constrained. Lucas and "
        "Kanade [1] resolve this by assuming neighboring pixels in a small window share "
        "approximately the same motion."
    )

    st.subheader("4. Lucas-Kanade: overdetermined system and least squares")
    st.markdown(
        "Stacking the constraint equation over `n` pixels `p_1, ..., p_n` in a window gives "
        "an overdetermined linear system `A*v = b`:"
    )
    st.latex(
        r"A = \begin{bmatrix} I_x(p_1) & I_y(p_1) \\ I_x(p_2) & I_y(p_2) \\ "
        r"\vdots & \vdots \\ I_x(p_n) & I_y(p_n) \end{bmatrix}, \qquad "
        r"b = \begin{bmatrix} -I_t(p_1) \\ -I_t(p_2) \\ \vdots \\ -I_t(p_n) \end{bmatrix}"
    )
    st.markdown("Solved by least squares via the normal equations:")
    st.latex(r"A^T A\, v = A^T b \qquad\Longrightarrow\qquad v = (A^T A)^{-1} A^T b")
    st.markdown(
        "valid when `A^T A` (the structure tensor) is invertible - i.e. both eigenvalues are "
        "large enough. This is the same criterion `cv2.goodFeaturesToTrack` uses to select "
        "trackable corners and that `calcOpticalFlowPyrLK`'s `minEigThreshold` uses to reject "
        "unreliable tracks (Phase 2, `module5_6.tracking`). OpenCV's implementation is "
        "pyramidal and iterative - it does not execute the equations above literally line by "
        "line, but solves this same least-squares model at each pyramid level. Full "
        "explanation: docs/MOTION_TRACKING_DERIVATION.md."
    )

    st.subheader("5. Bilinear interpolation: derivation from two 1D interpolations")
    st.markdown(
        "A tracked point's location is generally subpixel/fractional, not aligned to the "
        "integer pixel grid. Bilinear interpolation estimates a value there from its four "
        "integer-pixel neighbors `I00, I10, I01, I11`, with fractional offsets "
        "`α = x - x0` and `β = y - y0`. Interpolating along `x` at each known row, "
        "then along `y` between those two results (full derivation: "
        "docs/BILINEAR_INTERPOLATION.md):"
    )
    st.latex(r"R_0 = (1-\alpha) I_{00} + \alpha I_{10}, \qquad R_1 = (1-\alpha) I_{01} + \alpha I_{11}")
    st.latex(r"I(x,y) = (1-\beta) R_0 + \beta R_1")
    st.markdown("which expands to the final weighted expression:")
    st.latex(
        r"I(x,y) = (1-\alpha)(1-\beta) I_{00} + \alpha(1-\beta) I_{10} "
        r"+ (1-\alpha)\beta I_{01} + \alpha\beta I_{11}"
    )

    st.subheader("6. Interactive bilinear interpolation demonstration")
    st.caption(
        "Configure the four neighboring pixel intensities and the fractional coordinate; the "
        "weights and interpolated value below are computed live by "
        "module5_6.interpolation.bilinear_interpolate_corners - the same function the tests "
        "and this page's derivation both rely on. This is a mathematical demonstration, not "
        "assignment experimental evidence."
    )
    c1, c2 = st.columns(2)
    with c1:
        i00 = st.number_input("I00 (top-left)", value=10.0, step=1.0, key="theory_i00")
        i01 = st.number_input("I01 (bottom-left)", value=30.0, step=1.0, key="theory_i01")
    with c2:
        i10 = st.number_input("I10 (top-right)", value=20.0, step=1.0, key="theory_i10")
        i11 = st.number_input("I11 (bottom-right)", value=40.0, step=1.0, key="theory_i11")
    alpha = float(st.slider("alpha = x - x0", 0.0, 1.0, 0.3, step=0.01, key="theory_alpha"))
    beta = float(st.slider("beta = y - y0", 0.0, 1.0, 0.7, step=0.01, key="theory_beta"))

    weight_00, weight_10, weight_01, weight_11 = bilinear_weights(alpha, beta)
    value = bilinear_interpolate_corners(i00, i10, i01, i11, alpha, beta)

    w1, w2, w3, w4, w5 = st.columns(5)
    w1.metric("w00", f"{weight_00:.3f}")
    w2.metric("w10", f"{weight_10:.3f}")
    w3.metric("w01", f"{weight_01:.3f}")
    w4.metric("w11", f"{weight_11:.3f}")
    w5.metric("I(x, y)", f"{value:.3f}")
    st.caption(f"Weights sum to {weight_00 + weight_10 + weight_01 + weight_11:.6f} (expected 1.0).")

    st.pyplot(_bilinear_diagram(i00, i10, i01, i11, alpha, beta))
    st.caption(
        "Default values (I00=10, I10=20, I01=30, I11=40, alpha=0.3, beta=0.7) reproduce the "
        "worked example in docs/BILINEAR_INTERPOLATION.md Section 4 (expected result: 27.0)."
    )


def _sfm_page() -> None:
    foundation_page(
        "Structure From Motion",
        "Question 2: four-viewpoint image upload, feature correspondence, planar homography "
        "estimation, reprojection validation, and recovered-boundary visualization will be "
        "added in a later phase.",
    )


def _experiments_page() -> None:
    st.header("Experiments & Results")
    st.info(
        "Question 1's professor-required two-consecutive-frame pixel-location validation "
        "(IMPLEMENTATION_PLAN.md Section 12) and consolidated video/experiment evidence. "
        "Structure-from-motion experiment results are a later phase."
    )

    st.subheader("Video status")
    video_1_path = find_supplied_video(_REPO_ROOT / "data" / "videos" / "video_1")
    video_2_path = find_supplied_video(_REPO_ROOT / "data" / "videos" / "video_2")
    c1, c2 = st.columns(2)
    with c1:
        if video_1_path is not None:
            st.success(f"Video 1 supplied: {video_1_path.name}")
        else:
            st.warning("Video 1: PENDING USER EXPERIMENT - not yet supplied in data/videos/video_1/")
    with c2:
        if video_2_path is not None:
            st.success(f"Video 2 supplied: {video_2_path.name}")
        else:
            st.warning("Video 2: PENDING USER EXPERIMENT - not yet supplied in data/videos/video_2/")

    if video_1_path is None and video_2_path is None:
        pending_experiment_banner(
            "Neither assignment video has been supplied yet. Add real video files under "
            "data/videos/video_1/ and data/videos/video_2/, or point "
            "scripts/process_optical_flow.py / scripts/validate_tracking.py at your own copy "
            "of them, then reload this page. Do not substitute a synthetic or unrelated video "
            "for the assignment submission."
        )
    manifest = default_experiment_manifest()
    st.caption(
        f"Experiment manifest schema version {manifest['schema_version']} "
        "(data/experiment_manifest.json documents the full per-video schema; see "
        "docs/EXPERIMENTAL_RESULTS.md)."
    )

    st.subheader("Optical-flow evidence")
    st.caption(
        "Run scripts/process_optical_flow.py against each supplied video to generate the "
        "optical-flow visualization video and magnitude/direction summary required by "
        "Question 1; results are written under results/optical_flow/<video_id>/."
    )
    for video_id in ("video_1", "video_2"):
        summary_path = _REPO_ROOT / "results" / "optical_flow" / video_id / f"{video_id}_optical_flow_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
                st.success(f"{video_id}: optical-flow evidence available ({summary_path}).")
                st.json(summary)
            except (ValueError, OSError) as exc:
                st.error(f"Could not read {summary_path}: {exc}")
        else:
            st.warning(f"{video_id}: optical-flow evidence PENDING USER EXPERIMENT ({summary_path} not found).")

    st.subheader("Two-consecutive-frame pixel-location validation records")
    st.caption(
        "Upload one or more validation-record JSON files produced by the Motion Tracking "
        "page's manual-validation workflow or by scripts/validate_tracking.py. The table "
        "below reproduces IMPLEMENTATION_PLAN.md Section 12's required table shape."
    )
    uploads = st.file_uploader(
        "Validation record JSON files", type=["json"], accept_multiple_files=True, key="experiments_records"
    )
    records: list[TrackingValidationRecord] = []
    for upload in uploads or []:
        try:
            data = json.loads(upload.getvalue().decode("utf-8"))
            records.append(TrackingValidationRecord.from_dict(data))
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Could not parse {upload.name}: {exc}")

    if not records:
        st.table(
            [
                {
                    "Video": video,
                    "Point": point,
                    "Frame 1": "Pending",
                    "Predicted Frame 2": "Pending",
                    "Observed Frame 2": "Pending",
                    "Pixel Error": "Pending",
                }
                for video in ("Video 1", "Video 2")
                for point in ("P1", "P2")
            ]
        )
        pending_experiment_banner(
            "No validation records uploaded yet. This table mirrors "
            "IMPLEMENTATION_PLAN.md Section 12's required shape until real records exist - "
            "PENDING USER EXPERIMENT: real videos not yet supplied."
        )
    else:
        st.dataframe(records_to_table(records), width="stretch")
        completed = [record for record in records if record.pixel_error is not None]
        if completed:
            errors = [record.pixel_error for record in completed]
            s1, s2, s3 = st.columns(3)
            s1.metric("Completed records", len(completed))
            s2.metric("Mean pixel error (px)", f"{sum(errors) / len(errors):.3f}")
            s3.metric("Max pixel error (px)", f"{max(errors):.3f}")
        pending_count = len(records) - len(completed)
        if pending_count:
            st.warning(f"{pending_count} of {len(records)} uploaded record(s) are still awaiting manual observation.")
        st.caption(
            "Pixel error above is the professor-required comparison between the predicted and "
            "manually observed Frame 2 location - distinct from OpenCV's algorithmic tracking "
            "error and forward-backward consistency shown on the Motion Tracking page. Do not "
            "overstate what a small error means; it reflects only the point(s) actually "
            "measured."
        )

    st.subheader("Structure From Motion")
    st.write(
        "Four-view SfM experiment results, camera information, and reprojection results will "
        "be added in a later approved phase."
    )
    pending_experiment_banner(
        "This section is structurally available now. Its computer-vision processing is "
        "scheduled for a later approved phase."
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
