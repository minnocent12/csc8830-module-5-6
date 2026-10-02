"""Module 5-6 Streamlit pages and the get_pages provider.

Optical Flow, Motion Tracking, and Bilinear Interpolation & Theory implement Question 1. The
Structure From Motion page (Question 2) defaults to showing the completed real four-view
experiment from results/sfm/sfm_summary.json when no images are uploaded (see
`_render_sfm_bundled_results`), and otherwise runs the same live ORB/homography pipeline on
whatever planar-object images are uploaded. Experiments & Results consolidates both questions'
real results. Any page whose real data is genuinely not yet available falls back to an honest
pending-safe placeholder rather than fabricating a result (see IMPLEMENTATION_PLAN.md).
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: Streamlit's script runner executes pages on a
# worker thread, and matplotlib's platform-default interactive backend (e.g. macOS's "macosx")
# cannot create a GUI figure manager off the main thread and raises RuntimeError - Agg has no
# such restriction and is the standard choice for any web-server plotting use case.
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from module5_6.experiment import (
    TrackingValidationRecord,
    default_experiment_manifest,
    draw_validation_overlay,
    find_supplied_video,
    load_experiment_manifest,
    record_observation,
    records_to_table,
)
from module5_6.camera import ViewMetadata
from module5_6.features import ORBParams, detect_and_describe
from module5_6.geometry import boundary_polygon_closed
from module5_6.homography import HomographyParams
from module5_6.interpolation import bilinear_interpolate_corners, bilinear_weights
from module5_6.io_utils import decode_image_bgr, to_grayscale
from module5_6.sfm import register_view
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
from module5_6.webapp.ui import IMAGE_TYPES, VIDEO_TYPES, pending_experiment_banner
from module5_6.webapp.design.components import (
    ImageItem,
    configuration_card,
    data_table,
    download_action,
    image_card,
    image_comparison,
    metric_card,
    metric_row,
    page_header,
    parameter_group,
    result_section,
    section_header,
    upload_panel,
)

_MODULE = "Module 5-6"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OPTICAL_FLOW_RESULTS_DIR = _REPO_ROOT / "results" / "optical_flow"
_TRACKING_RESULTS_DIR = _REPO_ROOT / "results" / "tracking"


def _public_record(value):
    """Copy of a saved record for display, with file paths reduced to file names.

    The committed record is unchanged; only the displayed copy drops repository folders
    (``data/videos/video_1/IMG_7272.MOV`` is shown as ``IMG_7272.MOV``).
    """
    if isinstance(value, dict):
        return {key: _public_record(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_record(item) for item in value]
    if isinstance(value, str) and "/" in value and "." in value.rsplit("/", 1)[-1]:
        return value.rsplit("/", 1)[-1]
    return value


def _bgr_to_rgb(image):
    """Reverse the channel order for display with st.image; never mutates the input."""
    return image[:, :, ::-1]


def _draw_boundary_overlay(frame_bgr, boundary_points, *, color=(0, 0, 255), thickness=2):
    """Draw a closed boundary polygon on a copy of frame_bgr; never mutates the input."""
    overlay = frame_bgr.copy()
    closed = boundary_polygon_closed(boundary_points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [closed], isClosed=False, color=color, thickness=thickness)
    for x, y in boundary_points:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 5, color, -1)
    return overlay


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


def _render_optical_flow_bundled_results() -> bool:
    """Render committed real optical-flow evidence for Video 1/2, if available.

    Reads only results/optical_flow/<video_id>/ (committed JSON summary + evidence PNGs) -
    never the gitignored raw source video - so this renders identically on a fresh clone or
    the public deployment (root AGENTS.md "Bundled real-sample fallback"). Returns True if at
    least one video's evidence was rendered.
    """
    rendered_any = False
    for video_id in ("video_1", "video_2"):
        result_dir = _OPTICAL_FLOW_RESULTS_DIR / video_id
        summary_path = result_dir / f"{video_id}_optical_flow_summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (ValueError, OSError) as exc:
            st.error(f"Could not read {summary_path}: {exc}")
            continue
        rendered_any = True
        st.subheader(f"{video_id}: {Path(summary['source_video']).name}")
        st.caption(
            f"{summary['width']}x{summary['height']} px | {summary['fps']:.2f} fps | "
            f"sample frames {summary['sample_start_frame']}-{summary['sample_end_frame']} "
            f"({summary['sample_duration_seconds']:.1f} s, {summary['frame_pairs']} frame pairs)"
        )
        original = sorted(result_dir.glob(f"{video_id}_evidence_original_frame*.png"))
        hsv_flow = sorted(result_dir.glob(f"{video_id}_evidence_hsv_flow_*.png"))
        arrows = sorted(result_dir.glob(f"{video_id}_evidence_arrows_*.png"))
        c1, c2, c3 = st.columns(3)
        with c1:
            if original:
                st.image(str(original[0]), caption="Original frame", width="stretch")
        with c2:
            if hsv_flow:
                st.image(str(hsv_flow[0]), caption="HSV flow: hue = direction, value = magnitude", width="stretch")
        with c3:
            if arrows:
                st.image(str(arrows[0]), caption="Vector/arrow overlay", width="stretch")
        s1, s2, s3 = st.columns(3)
        s1.metric("Mean |flow| (px)", f"{summary['mean_magnitude']:.3f}")
        s2.metric("Median |flow| (px)", f"{summary['median_magnitude']:.3f}")
        s3.metric("Max |flow| (px)", f"{summary['max_magnitude']:.3f}")
    if rendered_any:
        st.caption(
            "These saved results cover the full required sample of both assignment videos; "
            "the statistics describe apparent motion in each sample, not object identity or "
            "scene depth."
        )
    return rendered_any


def _render_tracking_bundled_results() -> bool:
    """Render the committed real two-consecutive-frame validation records for Video 1/2.

    Reads only results/tracking/<video_id>/ (committed JSON record + evidence PNGs) - never
    the gitignored raw source video - so this renders identically on a fresh clone or the
    public deployment (root AGENTS.md "Bundled real-sample fallback").
    """
    completed = []
    for video_id in ("video_1", "video_2"):
        result_dir = _TRACKING_RESULTS_DIR / video_id
        record_paths = sorted(result_dir.glob("*_record.json")) if result_dir.is_dir() else []
        for record_path in record_paths:
            try:
                record = TrackingValidationRecord.from_dict(json.loads(record_path.read_text()))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                st.error(f"Could not read {record_path}: {exc}")
                continue
            if record.pixel_error is None:
                continue
            completed.append((video_id, result_dir, record))
    if not completed:
        return False

    with result_section(
        "Committed Tracking Evidence",
        description=(
            "Saved validation records and evidence images for the two assignment videos "
            "(not a live run)."
        ),
    ):
        for video_id, result_dir, record in completed:
            st.markdown(f"#### {video_id}: point {record.point_label}")
            frame1_path = result_dir / f"{record.point_label}_frame1.png"
            validated_path = result_dir / f"{record.point_label}_frame2_validated.png"
            frame1 = ImageItem(
                str(frame1_path),
                caption=f"Frame {record.frame1_index}: P = ({record.x1:.1f}, {record.y1:.1f})",
            )
            frame2 = ImageItem(
                str(validated_path),
                caption=f"Frame {record.frame2_index}: predicted (red) vs. observed (green)",
            )
            if frame1_path.is_file() and validated_path.is_file():
                image_comparison(frame1, frame2, bordered=False)
            else:  # show whichever committed image exists, in its usual column
                for column, path, item in zip(
                    st.columns(2), (frame1_path, validated_path), (frame1, frame2)
                ):
                    if path.is_file():
                        with column:
                            image_card(item.image, caption=item.caption, bordered=False)
            metric_row(
                [
                    ("Predicted Frame 2", f"({record.predicted_x2:.1f}, {record.predicted_y2:.1f})"),
                    ("Observed Frame 2 (manual)", f"({record.observed_x:.1f}, {record.observed_y:.1f})"),
                    ("Pixel error e (px)", f"{record.pixel_error:.3f}"),
                ]
            )
            st.caption(f"Observation method: {record.observation_method}")

    section_header("Interpretation")
    st.caption(
        "'Pixel error' compares the algorithmic Lucas-Kanade prediction with the Frame 2 "
        "location identified by visual inspection. It is the two-frame pixel-location "
        "validation for each assignment video."
    )
    return True


def _optical_flow_page() -> None:
    st.header("Optical Flow")
    st.info(
        "Question 1: upload a video to compute dense Farneback optical flow over a sampled "
        "interval and visualize it as a video. Sparse Lucas-Kanade point tracking is on the "
        "Motion Tracking page."
    )
    upload = st.file_uploader("Video", type=VIDEO_TYPES)
    if upload is None:
        has_bundled = any(
            (_OPTICAL_FLOW_RESULTS_DIR / v / f"{v}_optical_flow_summary.json").is_file()
            for v in ("video_1", "video_2")
        )
        if has_bundled:
            st.info(
                "No video uploaded, so this shows the completed real optical-flow evidence for "
                "both required assignment videos (committed results, not a placeholder). "
                "Upload your own video above to run the live pipeline on it instead; that "
                "overrides this view."
            )
            _render_optical_flow_bundled_results()
        else:
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
                    "Bounds compute time for this live demo. The submitted results process the full "
                    "required sample (shown when no video is uploaded)."
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
                f"{duration_seconds:.1f} s sample. This cap affects only this live demo; the "
                "submitted results cover the full sample."
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
            "are exploratory tooling for that upload, not the assignment's required two-frame "
            "pixel tracking validation (see the Motion Tracking page for that, including the "
            "completed result for the two assignment videos)."
        )
    finally:
        input_path.unlink(missing_ok=True)


_TRACKING_RUN_KEY = "tracking_completed_run"


@dataclass(frozen=True)
class _TrackingRun:
    """What a completed Motion Tracking run needs to be redisplayed and validated.

    Kept in ``st.session_state`` so the manual-validation widgets, whose changes rerun the
    script with the run button released, keep working on the same result.
    """

    start_frame: int
    frame_count: int
    feature_count: int
    features_image: np.ndarray  # display-ready RGB
    tracked_image: np.ndarray
    displacement_image: np.ndarray | None
    trajectory_image: np.ndarray
    trajectory_count: int
    alive_full_length: int
    valid_previous: np.ndarray
    valid_next: np.ndarray
    valid_error: np.ndarray
    valid_fb_error: np.ndarray | None
    frame2_bgr: np.ndarray  # source Frame 2 for the validation crop and overlay


def _tracking_signature(
    upload,
    start_seconds: float,
    frames_to_load: int,
    shi_tomasi_params: ShiTomasiParams,
    lk_params: LucasKanadeParams,
    validate_fb: bool,
    max_fb_error: float,
) -> tuple:
    """Everything that determines a tracking result; manual-validation inputs are excluded.

    The upload is identified by Streamlit's per-upload ``file_id`` (new for every upload,
    even of a same-named file) plus name and size, so the video is never re-hashed.
    """
    return (
        (upload.file_id, upload.name, upload.size),
        start_seconds,
        frames_to_load,
        repr(shi_tomasi_params),
        repr(lk_params),
        validate_fb,
        max_fb_error,
    )


def _compute_tracking_run(
    input_path: Path,
    metadata,
    start_seconds: float,
    frames_to_load: int,
    shi_tomasi_params: ShiTomasiParams,
    lk_params: LucasKanadeParams,
    validate_fb: bool,
    max_fb_error: float,
) -> _TrackingRun | None:
    """Run detection and tracking exactly as before; ``None`` after showing why it could not run.

    Called only when "Detect and track features" is clicked. Drawn evidence images are made
    here once, with the same calls as before, so later reruns only redisplay them.
    """
    start_frame = int(round(start_seconds * metadata.fps))
    start_frame = max(0, min(start_frame, max(0, metadata.frame_count - 2)))
    end_frame = min(metadata.frame_count, start_frame + frames_to_load)

    try:
        frames = read_frame_range(input_path, start_frame, end_frame)
    except ValueError as exc:
        st.error(f"Could not read the requested frames: {exc}")
        return None

    if len(frames) < 2:
        st.error("At least two frames are required to track features; choose an earlier start time.")
        return None

    grays = [to_grayscale(frame) for frame in frames]
    features = detect_features(grays[0], params=shi_tomasi_params)
    if features.shape[0] == 0:
        st.warning(
            "No Shi-Tomasi features were detected on Frame 1. Try a lower quality level or "
            "a smaller min-distance."
        )
        return None

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

    trajectories = track_trajectories(grays, shi_tomasi_params=shi_tomasi_params, lk_params=lk_params)
    alive_full_length = sum(1 for t in trajectories if len(t.positions) == len(grays))
    return _TrackingRun(
        start_frame=start_frame,
        frame_count=len(frames),
        feature_count=int(features.shape[0]),
        features_image=_bgr_to_rgb(draw_tracked_points(frames[0], features, color=(0, 255, 0))),
        tracked_image=_bgr_to_rgb(draw_tracked_points(frames[1], valid_next, color=(0, 0, 255))),
        displacement_image=(
            _bgr_to_rgb(draw_displacement_vectors(frames[0], valid_previous, valid_next))
            if valid_previous.shape[0] > 0
            else None
        ),
        trajectory_image=_bgr_to_rgb(draw_trajectories(frames[-1], trajectories)),
        trajectory_count=len(trajectories),
        alive_full_length=alive_full_length,
        valid_previous=valid_previous,
        valid_next=valid_next,
        valid_error=valid_error,
        valid_fb_error=valid_fb_error,
        frame2_bgr=frames[1],
    )


def _render_tracking_run(run: _TrackingRun) -> None:
    """Show a completed run and the manual validation; reruns redisplay the stored run."""
    start_frame = run.start_frame
    valid_previous, valid_next = run.valid_previous, run.valid_next
    valid_error, valid_fb_error = run.valid_error, run.valid_fb_error

    section_header("Tracking Evidence")
    image_comparison(
        ImageItem(
            run.features_image,
            caption=f"Frame {start_frame}: detected Shi-Tomasi features ({run.feature_count})",
        ),
        ImageItem(
            run.tracked_image,
            caption=f"Frame {start_frame + 1}: tracked features ({valid_next.shape[0]} valid)",
        ),
        bordered=False,
    )

    if valid_previous.shape[0] == 0:
        st.warning(
            "No tracks passed validation. Try relaxing the forward-backward threshold or "
            "adjusting the Lucas-Kanade window size."
        )

    # Displacement vectors and track history side by side: full-width portrait frames
    # would otherwise each fill several screens.
    displacement_column, history_column = st.columns(2)
    if valid_previous.shape[0] > 0:
        with displacement_column:
            image_card(
                run.displacement_image,
                title="Displacement vectors (Frame 1 -> Frame 2)",
                caption="Displacement vector (u, v) for each valid tracked point",
                bordered=False,
            )
    with history_column:
        image_card(
            run.trajectory_image,
            title="Track history across the loaded frames",
            caption=(
                f"Trajectories over {run.frame_count} frames (frames {start_frame} - "
                f"{start_frame + run.frame_count - 1}); {run.alive_full_length} of {run.trajectory_count} "
                "tracked points survived the full window"
            ),
            bordered=False,
        )

    if valid_previous.shape[0] > 0:
        displacements = valid_next - valid_previous
        magnitudes = displacement_magnitude(displacements)

        section_header("Selected Statistics")
        metric_row(
            [
                ("Detected features", run.feature_count),
                ("Valid tracks", valid_previous.shape[0]),
                ("Mean |displacement| (px)", f"{float(magnitudes.mean()):.3f}"),
                ("Max |displacement| (px)", f"{float(magnitudes.max()):.3f}"),
            ]
        )

        section_header("Tracking Data", description="Point coordinates and tracking-quality metrics")
        st.caption(
            "'LK consistency error' and 'Forward-backward consistency (px)' below are "
            "algorithmic self-consistency signals reported by this run's Lucas-Kanade "
            "tracker - they measure how well the tracker's own model fit, and how well a "
            "point tracked forward then backward returns to where it started. They are "
            "**not** the assignment's required pixel-location validation, which compares a "
            "predicted location against an actual observed location manually identified in "
            "a real video frame - that comparison is below, on this uploaded video "
            "(the completed result for the two assignment videos is shown when no video is "
            "uploaded)."
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
        data_table(rows)
        if valid_previous.shape[0] > max_rows:
            st.caption(f"Showing the first {max_rows} of {valid_previous.shape[0]} valid tracks.")
        st.caption(
            "Frame 2 coordinates above are the tracker's *predicted* location, not a "
            "manually observed one; no actual observed pixel location has been recorded "
            "for these points."
        )

        section_header(
            "Manual Validation",
            description="Two-frame pixel-location validation",
        )
        st.caption(
            "Select a point, inspect Frame 2, and enter the observed coordinates. The app will "
            "calculate the pixel error between the predicted and observed locations. Unlike "
            "'LK consistency error' and 'Forward-backward consistency' above, which never "
            "look at Frame 2 as an image, this pixel error is meaningful only if the observed "
            "coordinate truly comes from inspecting Frame 2, not from accepting a default or "
            "copying the prediction."
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
        frame2_height, frame2_width = run.frame2_bgr.shape[:2]
        x_lo = max(0, crop_center_x - crop_half)
        x_hi = min(frame2_width, crop_center_x + crop_half)
        y_lo = max(0, crop_center_y - crop_half)
        y_hi = min(frame2_height, crop_center_y + crop_half)
        crop = run.frame2_bgr[y_lo:y_hi, x_lo:x_hi]
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
            metric_card("Pixel error e (predicted vs. observed)", f"{validation_record.pixel_error:.3f} px")
            st.image(
                _bgr_to_rgb(draw_validation_overlay(run.frame2_bgr, validation_record)),
                caption="Predicted (red) vs. observed (green) Frame 2 location",
                width="stretch",
            )
            download_action(
                "Download validation record (JSON)",
                json.dumps(validation_record.to_dict(), indent=2),
                file_name=f"{video_id}_{point_label}_validation_record.json",
                mime="application/json",
                key="tracking_validation_download",
            )
            st.caption(
                "Upload this record on the Experiments & Results page to include it in the "
                "consolidated validation table. This pixel error is real and computed from the coordinates you "
                "entered - it is only meaningful evidence if the observed coordinate truly "
                "came from inspecting Frame 2."
            )
        else:
            st.warning(
                "Manual pixel-location validation for this point is pending until you inspect "
                "Frame 2 above, enter the actual observed coordinate, and check the "
                "confirmation box."
            )

    section_header("Interpretation")
    st.caption(
        "These trajectories are computed live from whatever video was uploaded above; they "
        "are exploratory tooling, not the assignment's required experimental evidence. The "
        "validation results for the assignment videos are available on this page when no "
        "video is uploaded."
    )


_MOTION_TRACKING_INTRO = (
    "Question 1: Shi-Tomasi feature detection plus pyramidal Lucas-Kanade tracking between "
    "consecutive frames, solving the two-frame tracking problem (find P' = (x+u, y+v) in "
    "Frame 2 for each point P = (x, y) in Frame 1). The formal brightness-constancy/"
    "Lucas-Kanade derivation is on the Bilinear Interpolation & Theory page; the manual "
    "pixel-location validation is below, once features are detected and tracked."
)


def _motion_tracking_page() -> None:
    page_header("Motion Tracking", eyebrow=_MODULE, description=_MOTION_TRACKING_INTRO)

    section_header("Input")
    upload = upload_panel("Video", type=VIDEO_TYPES, key="tracking_upload")
    if upload is None:
        has_bundled = any(
            list((_TRACKING_RESULTS_DIR / v).glob("*_record.json"))
            for v in ("video_1", "video_2")
            if (_TRACKING_RESULTS_DIR / v).is_dir()
        )
        if has_bundled:
            st.info(
                "No video uploaded, so this shows the completed real two-consecutive-frame "
                "pixel-location validation for both required assignment videos (committed "
                "results, not a placeholder). Upload your own video above to run the live "
                "detection/tracking/validation pipeline instead; that overrides this view."
            )
            _render_tracking_bundled_results()
        else:
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

        with configuration_card():
            with parameter_group("Frames"):
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

            run = st.button("Detect and track features", type="primary")

        signature = _tracking_signature(
            upload, start_seconds, frames_to_load, shi_tomasi_params, lk_params, validate_fb, max_fb_error
        )
        if run:
            tracking_run = _compute_tracking_run(
                input_path, metadata, start_seconds, frames_to_load,
                shi_tomasi_params, lk_params, validate_fb, max_fb_error,
            )
            if tracking_run is None:
                st.session_state.pop(_TRACKING_RUN_KEY, None)
                return
            st.session_state[_TRACKING_RUN_KEY] = (signature, tracking_run)
        else:
            stored = st.session_state.get(_TRACKING_RUN_KEY)
            if stored is None or stored[0] != signature:
                if stored is not None:  # never show a result against settings it was not run with
                    del st.session_state[_TRACKING_RUN_KEY]
                    st.info(
                        "The video or tracking settings changed since the last run, so its "
                        "results are no longer shown. Select Detect and track features to run "
                        "the experiment again."
                    )
                else:
                    st.info(
                        "Set the frame window and the Shi-Tomasi and Lucas-Kanade settings, then "
                        "select Detect and track features to run the experiment."
                    )
                return
            tracking_run = stored[1]
        _render_tracking_run(tracking_run)
    finally:
        input_path.unlink(missing_ok=True)


def _theory_page() -> None:
    st.header("Bilinear Interpolation & Theory")
    st.info(
        "Question 1's required theory: brightness constancy through the optical-flow "
        "constraint equation and the aperture problem, the Lucas-Kanade least-squares "
        "tracking derivation and its relationship to the OpenCV implementation used on the "
        "Motion Tracking page, and the bilinear-interpolation derivation with an "
        "interactive demonstration."
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
    st.caption("The full step-by-step derivation is included in the written report.")

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
        "unreliable tracks on the Motion Tracking page. OpenCV's implementation is "
        "pyramidal and iterative - it does not execute the equations above literally line by "
        "line, but solves this same least-squares model at each pyramid level."
    )

    st.subheader("5. Bilinear interpolation: derivation from two 1D interpolations")
    st.markdown(
        "A tracked point's location is generally subpixel/fractional, not aligned to the "
        "integer pixel grid. Bilinear interpolation estimates a value there from its four "
        "integer-pixel neighbors `I00, I10, I01, I11`, with fractional offsets "
        "`α = x - x0` and `β = y - y0`. Interpolating along `x` at each known row, "
        "then along `y` between those two results:"
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
        "worked example from the written derivation (expected result: 27.0)."
    )


_SFM_SUMMARY_PATH = _REPO_ROOT / "results" / "sfm" / "sfm_summary.json"


def _render_sfm_bundled_results(summary: dict) -> None:
    """Render the completed real Phase 6 four-view experiment from results/sfm/sfm_summary.json.

    Reads only the committed summary JSON and JPEG figures under results/sfm/ - never the
    original data/sfm/view_N/ photos, which stay gitignored (root AGENTS.md "Bundled
    real-sample fallback"), so this renders the same on a fresh clone as it does locally.
    """
    st.caption(summary.get("terminology_note", ""))
    st.write(summary.get("object_description", ""))

    reference_view_id = summary["reference_view_id"]
    st.write(f"**Reference view:** {reference_view_id}. {summary.get('reference_view_choice_rationale', '')}")

    st.subheader("Camera / view information (real)")
    rows = []
    for view_id, view in summary["views"].items():
        exif = view.get("real_exif", {})
        notes = view.get("capture_notes_user_recorded", {})
        device = " ".join(str(exif[k]) for k in ("Make", "Model") if k in exif) or "Unknown"
        rows.append(
            {
                "View": view_id,
                "File": Path(view["file"]).name,
                "Size (px)": f"{view['width']}x{view['height']}",
                "Device": device,
                "Focal length (mm)": exif.get("FocalLength"),
                "f-number": exif.get("FNumber"),
                "ISO": exif.get("ISOSpeedRatings"),
                "Approx. position": notes.get("orientation"),
                "Approx. distance": notes.get("distance_to_object_notes"),
            }
        )
    st.dataframe(rows, width="stretch")
    reference_kp_path = _REPO_ROOT / f"results/sfm/{reference_view_id}_orb_keypoints.jpg"
    if reference_kp_path.is_file():
        st.image(str(reference_kp_path), caption=f"{reference_view_id}: real ORB keypoints", width="stretch")

    for view_id, registration in summary["registrations"].items():
        st.subheader(f"{view_id} -> {reference_view_id}")
        reproj = registration["reprojection_error_px"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Retained matches", registration["retained_match_count"])
        s2.metric("RANSAC inliers", registration["inlier_count"])
        s3.metric("Inlier ratio", f"{100 * registration['inlier_ratio']:.1f}%")
        s4.metric("Mean reproj. error, inliers (px)", f"{reproj['mean_inliers']:.3f}")
        st.caption(
            f"Candidate matches: {registration['candidate_match_count']} -> "
            f"retained (Lowe's ratio test, threshold "
            f"{registration['match_params']['ratio_test_threshold']}): "
            f"{registration['retained_match_count']} -> RANSAC inliers: {registration['inlier_count']}. "
            f"Median reproj. error (inliers): {reproj['median_inliers']:.3f} px; "
            f"max: {reproj['max_inliers']:.3f} px."
        )
        c1, c2 = st.columns(2)
        with c1:
            matches_path = _REPO_ROOT / registration["artifacts"]["matches_inliers"]
            if matches_path.is_file():
                st.image(str(matches_path), caption=f"{view_id} | {reference_view_id}: RANSAC-inlier matches", width="stretch")
        with c2:
            warp_path = _REPO_ROOT / registration["artifacts"]["registered_warp"]
            if warp_path.is_file():
                st.image(str(warp_path), caption=f"{view_id} registered into {reference_view_id}'s frame", width="stretch")
        st.caption(f"Homography H({view_id} -> {reference_view_id}):")
        st.code(np.array2string(np.array(registration["homography_view_to_reference"]), precision=4, suppress_small=True))

    st.subheader("Boundary reconstruction")
    reconstruction_path = _REPO_ROOT / summary["boundary_reconstruction"]["artifacts"]["reference_boundary_reconstruction"]
    top_down_path = _REPO_ROOT / summary["boundary_reconstruction"]["artifacts"]["view_1_top_down_rectified"]
    c1, c2 = st.columns(2)
    with c1:
        if reconstruction_path.is_file():
            st.image(
                str(reconstruction_path),
                caption="Red: View 1's manual boundary. Yellow: consensus (registered) boundary.",
                width="stretch",
            )
    with c2:
        if top_down_path.is_file():
            st.image(str(top_down_path), caption="Normalized top-down rectification of View 1", width="stretch")

    workout = summary.get("mathematical_workout_real_data")
    if workout:
        st.subheader("Real mathematical workout (p' ~ H p)")
        st.write(
            f"Source point in {workout['source_view']}: "
            f"`{tuple(round(v, 3) for v in workout['p_view_xy'])}`. "
            f"Predicted (normalized `q = H p`) in {workout['reference_view']}: "
            f"`{tuple(round(v, 3) for v in workout['predicted_normalized_xy'])}`. "
            f"Actual observed point: `{tuple(round(v, 3) for v in workout['actual_observed_reference_xy'])}`. "
            f"Reprojection error: **{workout['reprojection_error_px']:.3f} px**."
        )
        st.caption("This calculation was also verified independently by hand.")

    st.caption(
        "The full write-up, including the camera geometry and the complete calculations, is "
        "in the written report."
    )


def _sfm_page() -> None:
    st.header("Structure From Motion")
    st.info(
        "Question 2: four-viewpoint planar homography registration for a flat/2D planar "
        "object - not a dense/full 3D reconstruction. Upload real images of a single planar object "
        "taken from different camera positions to run the live pipeline below, or leave empty "
        "to see the completed real four-view experiment."
    )

    st.subheader("View images")
    view_ids = ["view_1", "view_2", "view_3", "view_4"]
    images_bgr: dict[str, object] = {}
    image_names: dict[str, str] = {}
    columns = st.columns(4)
    for col, view_id in zip(columns, view_ids):
        with col:
            upload = st.file_uploader(view_id, type=IMAGE_TYPES, key=f"sfm_upload_{view_id}")
            if upload is not None:
                try:
                    images_bgr[view_id] = decode_image_bgr(upload.getvalue(), source_name=upload.name)
                    image_names[view_id] = upload.name
                    st.image(_bgr_to_rgb(images_bgr[view_id]), caption=upload.name, width="stretch")
                except (TypeError, ValueError) as exc:
                    st.error(f"Could not read {view_id}: {exc}")

    if len(images_bgr) < 2:
        if _SFM_SUMMARY_PATH.is_file():
            st.info(
                "No images uploaded, so this shows the completed real four-view SfM experiment "
                "(saved results, not a placeholder). "
                "Upload your own images above to run the live pipeline on a different object "
                "instead; that overrides this view."
            )
            try:
                bundled_summary = json.loads(_SFM_SUMMARY_PATH.read_text())
                _render_sfm_bundled_results(bundled_summary)
            except (ValueError, OSError) as exc:
                st.error(f"Could not load the bundled real SfM results: {exc}")
            return
        pending_experiment_banner(
            "Upload at least two views (ideally all four) to exercise feature matching and "
            "homography registration. The completed four-view experiment on the assignment "
            "object is shown when no images are uploaded."
        )
        return
    if len(images_bgr) < 4:
        st.warning(
            f"{len(images_bgr)} of the required 4 views are supplied - registration below runs "
            "on the uploaded views, but the assignment requires all four."
        )

    st.subheader("Camera / view metadata (optional)")
    st.caption(
        "Record any real camera information you actually have for each supplied view. "
        "Width/height come from the uploaded image itself; every other field stays blank "
        "(never fabricated) until you supply a real value."
    )
    view_metadata: dict[str, ViewMetadata] = {}
    for view_id in images_bgr:
        height, width = images_bgr[view_id].shape[:2]
        with st.expander(f"{view_id} metadata"):
            device = st.text_input("Camera/device", value="", key=f"sfm_{view_id}_device").strip() or None
            focal_length = (
                st.number_input(
                    "Focal length (mm) - leave 0 if unknown", min_value=0.0, value=0.0, key=f"sfm_{view_id}_focal"
                )
                or None
            )
            distance = (
                st.number_input(
                    "Distance to object (m) - leave 0 if unknown",
                    min_value=0.0,
                    value=0.0,
                    key=f"sfm_{view_id}_distance",
                )
                or None
            )
            orientation = st.text_input(
                "Approximate orientation (free text)", value="", key=f"sfm_{view_id}_orientation"
            ).strip() or None
            notes = st.text_area("Notes", value="", key=f"sfm_{view_id}_notes").strip() or None
            view_metadata[view_id] = ViewMetadata(
                view_id=view_id,
                image_path=image_names.get(view_id),
                width=width,
                height=height,
                device=device,
                focal_length_mm=focal_length,
                distance_to_object_m=distance,
                orientation=orientation,
                notes=notes,
                status="available",
            )
            st.json(view_metadata[view_id].to_dict())

    st.subheader("Reference view and registration settings")
    reference_view_id = st.selectbox("Reference view", options=list(images_bgr.keys()), key="sfm_reference_view")
    other_view_ids = [v for v in images_bgr if v != reference_view_id]

    with st.expander("Feature detection & homography settings", expanded=False):
        n_features = int(st.slider("ORB max features", 100, 2000, 500, step=100, key="sfm_n_features"))
        method_label = st.radio(
            "Homography method",
            ["RANSAC (robust, rejects outlier matches)", "All points (no outlier rejection)"],
            key="sfm_homography_method",
        )
        homography_method = "ransac" if method_label.startswith("RANSAC") else "all"
        ransac_threshold = float(
            st.slider("RANSAC reprojection threshold (px)", 1.0, 10.0, 3.0, key="sfm_ransac_threshold")
        )

    orb_params = ORBParams(n_features=n_features)
    homography_params = HomographyParams(method=homography_method, ransac_reproj_threshold=ransac_threshold)

    st.subheader("Object boundary (optional)")
    enable_boundary = st.checkbox(
        "Enter each view's four boundary corners to recover/register the object's boundary",
        value=False,
        key="sfm_enable_boundary",
    )
    boundary_points_by_view: dict[str, "np.ndarray"] = {}
    if enable_boundary:
        st.caption(
            "Pixel coordinates, top-left origin, x increases right, y increases down "
            "(the same convention used throughout this app)."
        )
        for view_id in images_bgr:
            height, width = images_bgr[view_id].shape[:2]
            default_points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
            with st.expander(f"{view_id} boundary corners", expanded=False):
                points = []
                corner_columns = st.columns(4)
                for index, corner_col in enumerate(corner_columns):
                    with corner_col:
                        x = st.number_input(
                            f"P{index + 1} x", 0, width - 1, default_points[index][0], key=f"sfm_{view_id}_p{index}_x"
                        )
                        y = st.number_input(
                            f"P{index + 1} y", 0, height - 1, default_points[index][1], key=f"sfm_{view_id}_p{index}_y"
                        )
                        points.append((float(x), float(y)))
                boundary_points_by_view[view_id] = np.array(points, dtype=np.float64)

    if not st.button("Detect features and register views", type="primary"):
        pending_experiment_banner(
            "Choose a reference view and settings above, then run to view detected features, "
            "matches, homography/reprojection statistics, and (if boundary corners were "
            "entered) the registered boundary overlay."
        )
        return

    reference_gray = to_grayscale(images_bgr[reference_view_id])
    reference_keypoints, reference_descriptors = detect_and_describe(reference_gray, params=orb_params)
    st.subheader(f"Reference view: {reference_view_id}")
    st.image(
        _bgr_to_rgb(draw_tracked_points(images_bgr[reference_view_id], reference_keypoints, color=(0, 255, 0))),
        caption=f"{reference_keypoints.shape[0]} ORB features detected",
        width="stretch",
    )
    if reference_keypoints.shape[0] == 0:
        st.error("No ORB features were detected on the reference view; try a more textured image or lower settings.")
        return

    for view_id in other_view_ids:
        st.subheader(f"{view_id} -> {reference_view_id}")
        view_gray = to_grayscale(images_bgr[view_id])
        try:
            registration = register_view(
                reference_keypoints,
                reference_descriptors,
                view_id,
                view_gray,
                orb_params=orb_params,
                homography_params=homography_params,
                boundary_points_view=boundary_points_by_view.get(view_id),
            )
        except ValueError as exc:
            st.error(f"Could not register {view_id}: {exc}")
            continue

        c1, c2 = st.columns(2)
        with c1:
            st.image(
                _bgr_to_rgb(
                    draw_tracked_points(images_bgr[view_id], registration.matched_points_view, color=(0, 0, 255))
                ),
                caption=f"{view_id}: {registration.matched_points_view.shape[0]} matched points",
                width="stretch",
            )
        with c2:
            st.image(
                _bgr_to_rgb(
                    draw_tracked_points(
                        images_bgr[reference_view_id], registration.matched_points_reference, color=(0, 255, 0)
                    )
                ),
                caption=f"{reference_view_id}: corresponding matched points",
                width="stretch",
            )

        inlier_count = int(registration.inlier_mask.sum())
        s1, s2, s3 = st.columns(3)
        s1.metric("Matches", registration.matched_points_view.shape[0])
        s2.metric("Inliers", inlier_count)
        s3.metric("Mean reprojection error, inliers (px)", f"{registration.mean_reprojection_error:.3f}")
        st.caption(
            "Homography H (view -> reference), estimated by "
            f"{'RANSAC' if homography_method == 'ransac' else 'all-point least squares'}:"
        )
        st.code(np.array2string(registration.homography, precision=4, suppress_small=True))

        if registration.registered_boundary is not None:
            st.image(
                _bgr_to_rgb(
                    _draw_boundary_overlay(images_bgr[reference_view_id], registration.registered_boundary)
                ),
                caption=f"{view_id}'s boundary corners, registered into {reference_view_id}'s frame",
                width="stretch",
            )

    st.caption(
        "These features, matches, homographies, and reprojection errors are computed live from "
        "whatever images were uploaded above; they are exploratory/software-verification "
        "tooling. The completed four-view experiment on the assignment object is shown when "
        "no images are uploaded."
    )


def _experiments_page() -> None:
    st.header("Experiments & Results")
    st.info(
        "Question 1's two-consecutive-frame pixel-location validation and consolidated "
        "video/experiment evidence, plus "
        "a summary of the completed Question 2 four-view Structure From Motion experiment."
    )

    st.subheader("Video status")
    st.caption(
        "Experiment status reflects the saved results for each video, not whether the large "
        "raw source video is present in this environment. The raw videos are not included "
        "in the app, but the completed evidence is still shown."
    )
    video_1_path = find_supplied_video(_REPO_ROOT / "data" / "videos" / "video_1")
    video_2_path = find_supplied_video(_REPO_ROOT / "data" / "videos" / "video_2")
    video_1_done = (_OPTICAL_FLOW_RESULTS_DIR / "video_1" / "video_1_optical_flow_summary.json").is_file()
    video_2_done = (_OPTICAL_FLOW_RESULTS_DIR / "video_2" / "video_2_optical_flow_summary.json").is_file()
    c1, c2 = st.columns(2)
    with c1:
        if video_1_done:
            st.success("Video 1: experiment complete.")
        else:
            st.warning("Video 1: PENDING USER EXPERIMENT - no saved optical-flow summary found.")
        st.caption(f"Raw source file present locally: {'yes (' + video_1_path.name + ')' if video_1_path else 'no'}")
    with c2:
        if video_2_done:
            st.success("Video 2: experiment complete.")
        else:
            st.warning("Video 2: PENDING USER EXPERIMENT - no saved optical-flow summary found.")
        st.caption(f"Raw source file present locally: {'yes (' + video_2_path.name + ')' if video_2_path else 'no'}")

    if not video_1_done and not video_2_done:
        pending_experiment_banner(
            "Neither assignment video's experiment has been completed yet. Process the two real "
            "assignment videos with the optical-flow and tracking-validation tools, then "
            "reload this page. Do not substitute a synthetic or unrelated video "
            "for the assignment submission."
        )
    manifest_path = _REPO_ROOT / "data" / "experiment_manifest.json"
    manifest = load_experiment_manifest(manifest_path) if manifest_path.is_file() else default_experiment_manifest()
    st.caption(
        f"Experiment manifest schema version {manifest['schema_version']}."
    )

    st.subheader("Optical-flow evidence")
    st.caption(
        "Each supplied video is processed into the optical-flow visualization video and "
        "magnitude/direction summary required by Question 1."
    )
    for video_id in ("video_1", "video_2"):
        summary_path = _REPO_ROOT / "results" / "optical_flow" / video_id / f"{video_id}_optical_flow_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
                st.success(f"{video_id}: optical-flow evidence available.")
                st.json(_public_record(summary))
            except (ValueError, OSError) as exc:
                st.error(f"Could not read {summary_path}: {exc}")
        else:
            st.warning(f"{video_id}: optical-flow evidence PENDING USER EXPERIMENT ({summary_path} not found).")

    st.subheader("Two-consecutive-frame pixel-location validation records")
    st.caption(
        "Saved validation records (from the tracking-validation tool or the Motion Tracking "
        "page's manual-validation workflow) are loaded automatically below; you can also "
        "upload additional record JSON files."
    )
    records: list[TrackingValidationRecord] = []
    seen_keys: set[tuple[str, str, int]] = set()
    tracking_dir = _REPO_ROOT / "results" / "tracking"
    if tracking_dir.is_dir():
        for record_path in sorted(tracking_dir.glob("*/*_record.json")):
            try:
                data = json.loads(record_path.read_text())
                record = TrackingValidationRecord.from_dict(data)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                st.error(f"Could not parse {record_path}: {exc}")
                continue
            key = (record.video_id, record.point_label, record.frame1_index)
            if key not in seen_keys:
                seen_keys.add(key)
                records.append(record)

    uploads = st.file_uploader(
        "Additional validation record JSON files", type=["json"], accept_multiple_files=True, key="experiments_records"
    )
    for upload in uploads or []:
        try:
            data = json.loads(upload.getvalue().decode("utf-8"))
            record = TrackingValidationRecord.from_dict(data)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Could not parse {upload.name}: {exc}")
            continue
        key = (record.video_id, record.point_label, record.frame1_index)
        if key not in seen_keys:
            seen_keys.add(key)
            records.append(record)

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
            "No validation records uploaded yet. This table shows the required record layout "
            "until real records exist - PENDING USER EXPERIMENT: real videos not yet "
            "supplied."
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
            "Pixel error above compares the predicted and "
            "manually observed Frame 2 location - distinct from OpenCV's algorithmic tracking "
            "error and forward-backward consistency shown on the Motion Tracking page. Do not "
            "overstate what a small error means; it reflects only the point(s) actually "
            "measured."
        )

    st.subheader("Structure From Motion")
    if _SFM_SUMMARY_PATH.is_file():
        try:
            sfm_summary = json.loads(_SFM_SUMMARY_PATH.read_text())
            st.success(
                f"Four-view SfM experiment complete - reference view "
                f"{sfm_summary['reference_view_id']}, "
                f"{len(sfm_summary['registrations'])} views registered. "
                "See the Structure From Motion page for the full real results."
            )
            reproj_means = [r["reprojection_error_px"]["mean_inliers"] for r in sfm_summary["registrations"].values()]
            inlier_counts = [r["inlier_count"] for r in sfm_summary["registrations"].values()]
            s1, s2 = st.columns(2)
            s1.metric("Views registered", len(sfm_summary["registrations"]))
            s2.metric(
                "Mean reproj. error range (px)",
                f"{min(reproj_means):.3f}-{max(reproj_means):.3f}",
            )
            st.caption(f"RANSAC inlier counts by view: {inlier_counts}")
        except (ValueError, KeyError, OSError) as exc:
            st.error(f"Could not read {_SFM_SUMMARY_PATH}: {exc}")
    else:
        st.write(
            "The reusable planar homography-registration foundation (ORB features, homography "
            "estimation/reprojection, boundary registration - see the Structure From Motion "
            "page) is implemented and tested against "
            "synthetic fixtures."
        )
        pending_experiment_banner(
            "No real four-view SfM results are available yet "
            f"({_SFM_SUMMARY_PATH} not found). PENDING USER EXPERIMENT."
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
