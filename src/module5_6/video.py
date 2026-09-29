"""Video loading, metadata, sample-interval extraction, and frame reading for Module 5-6.

This module owns all video file IO. Optical-flow computation itself lives in
``module5_6.optical_flow`` as pure array-in/array-out functions so it can be reused without a
video file (for example, on frames the web app already holds in memory).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from module5_6.io_utils import to_grayscale
from module5_6.optical_flow import FarnebackParams, compute_farneback_flow, draw_flow_arrows, flow_to_hsv_bgr
from module5_6.types import VideoMetadata

MINIMUM_SAMPLE_DURATION_SECONDS = 30.0


def open_video_capture(path: str | Path) -> cv2.VideoCapture:
    """Open a video file for reading, raising instead of returning an unusable capture."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video: {video_path}")
    return capture


def get_video_metadata(path: str | Path) -> VideoMetadata:
    """Read fps, frame count, dimensions, and duration without decoding any frame pixels."""
    video_path = Path(path)
    capture = open_video_capture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"video reports a non-positive fps: {video_path}")
    if frame_count <= 0:
        raise ValueError(f"video reports a non-positive frame count: {video_path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"video reports non-positive dimensions: {video_path}")
    return VideoMetadata(
        path=str(video_path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=frame_count / fps,
    )


def compute_sample_frame_range(
    metadata: VideoMetadata,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float = MINIMUM_SAMPLE_DURATION_SECONDS,
    minimum_duration_seconds: float | None = MINIMUM_SAMPLE_DURATION_SECONDS,
) -> tuple[int, int]:
    """Compute a half-open ``[start_frame, end_frame)`` range for a requested time interval.

    The Module 5-6 assignment (Question 1) requires at least a 30-second sample from each
    video; ``minimum_duration_seconds`` enforces that by default. Pass a smaller value (or
    ``None`` to disable the check entirely) only for deterministic tests against short
    synthetic clips or for an explicitly non-assignment interactive preview.
    """
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if minimum_duration_seconds is not None and duration_seconds < minimum_duration_seconds:
        raise ValueError(
            f"duration_seconds ({duration_seconds}) is below the required minimum "
            f"({minimum_duration_seconds}) seconds"
        )
    start_frame = int(round(start_seconds * metadata.fps))
    end_frame = int(round((start_seconds + duration_seconds) * metadata.fps))
    end_frame = min(end_frame, metadata.frame_count)
    if start_frame >= end_frame:
        raise ValueError(
            f"requested sample [{start_seconds}, {start_seconds + duration_seconds}) seconds "
            f"does not overlap the video duration (0, {metadata.duration_seconds:.3f}] seconds"
        )
    return start_frame, end_frame


def read_frame_range(path: str | Path, start_frame: int, end_frame: int) -> list[np.ndarray]:
    """Read BGR frames ``[start_frame, end_frame)`` in order, without mutating the source file."""
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")
    capture = open_video_capture(path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        frames: list[np.ndarray] = []
        for _ in range(end_frame - start_frame):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"no frames could be read in range [{start_frame}, {end_frame})")
    return frames


def open_video_writer(output_path: str | Path, *, fps: float, width: int, height: int) -> tuple[cv2.VideoWriter, Path]:
    """Open a browser-friendly H.264-in-MP4 writer, falling back to MPEG-4 if unavailable.

    Both fourccs were confirmed to round-trip (write then re-read) with this OpenCV build;
    ``avc1`` is preferred because Streamlit's ``st.video`` plays H.264 MP4 natively in the
    browser, while a plain ``mp4v`` stream sometimes does not.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_name in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, output_path
        writer.release()
    raise ValueError(f"could not open a video writer for: {output_path}")


def render_optical_flow_video(
    frames_bgr: Sequence[np.ndarray],
    output_path: str | Path,
    *,
    fps: float,
    mode: str = "hsv",
    farneback_params: FarnebackParams | None = None,
    arrow_step: int = 16,
    arrow_min_magnitude: float = 1.0,
) -> Path:
    """Compute consecutive-frame Farneback optical flow and write a visualization video.

    ``mode="hsv"`` writes the HSV-encoded flow-color video (hue = direction, value =
    magnitude); ``mode="arrows"`` writes the original frames with sampled flow-vector arrows
    overlaid. Requires at least two frames of identical shape.
    """
    if mode not in ("hsv", "arrows"):
        raise ValueError(f"unsupported mode: {mode!r}, expected 'hsv' or 'arrows'")
    if len(frames_bgr) < 2:
        raise ValueError("at least two frames are required to compute optical flow")
    height, width = frames_bgr[0].shape[:2]
    writer, output_path = open_video_writer(output_path, fps=fps, width=width, height=height)
    try:
        previous_gray = to_grayscale(frames_bgr[0])
        for frame in frames_bgr[1:]:
            current_gray = to_grayscale(frame)
            flow = compute_farneback_flow(previous_gray, current_gray, params=farneback_params)
            if mode == "hsv":
                visualization = flow_to_hsv_bgr(flow)
            else:
                visualization = draw_flow_arrows(
                    frame, flow, step=arrow_step, min_magnitude=arrow_min_magnitude
                )
            writer.write(visualization)
            previous_gray = current_gray
    finally:
        writer.release()
    return output_path
