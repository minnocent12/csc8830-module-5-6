from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from module5_6.types import VideoMetadata
from module5_6.video import (
    MINIMUM_SAMPLE_DURATION_SECONDS,
    compute_sample_frame_range,
    get_video_metadata,
    open_video_capture,
    read_frame_range,
    render_optical_flow_video,
)

_WIDTH, _HEIGHT = 32, 24


def _write_synthetic_video(path: Path, num_frames: int, *, fps: float = 10.0) -> Path:
    """Write a small, deterministic, lossless-enough synthetic video for round-trip tests.

    Frame i is a solid fill of value (i * 20) % 256 so frame identity/order can be checked
    after decoding without depending on any real assignment footage.
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (_WIDTH, _HEIGHT))
    assert writer.isOpened(), "synthetic test fixture writer failed to open"
    for index in range(num_frames):
        value = (index * 20) % 256
        writer.write(np.full((_HEIGHT, _WIDTH, 3), value, dtype=np.uint8))
    writer.release()
    return path


def _textured_frame(size: tuple[int, int] = (48, 48), *, offset: int = 0) -> np.ndarray:
    rng = np.random.default_rng(offset)
    gray = rng.integers(0, 256, size=size, dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_open_video_capture_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_video_capture(tmp_path / "does_not_exist.avi")


def test_get_video_metadata_reads_expected_properties(tmp_path: Path) -> None:
    video_path = _write_synthetic_video(tmp_path / "sample.avi", num_frames=20, fps=10.0)

    metadata = get_video_metadata(video_path)

    assert metadata.fps == pytest.approx(10.0, rel=0.05)
    assert metadata.frame_count == 20
    assert metadata.width == _WIDTH
    assert metadata.height == _HEIGHT
    assert metadata.duration_seconds == pytest.approx(2.0, rel=0.1)


def test_compute_sample_frame_range_basic() -> None:
    metadata = VideoMetadata(path="x", fps=10.0, frame_count=100, width=1, height=1, duration_seconds=10.0)

    start_frame, end_frame = compute_sample_frame_range(
        metadata, start_seconds=0.0, duration_seconds=5.0, minimum_duration_seconds=None
    )

    assert (start_frame, end_frame) == (0, 50)


def test_compute_sample_frame_range_enforces_assignment_minimum_by_default() -> None:
    metadata = VideoMetadata(path="x", fps=10.0, frame_count=100, width=1, height=1, duration_seconds=10.0)

    assert MINIMUM_SAMPLE_DURATION_SECONDS == 30.0
    with pytest.raises(ValueError, match="minimum"):
        compute_sample_frame_range(metadata, duration_seconds=5.0)


def test_compute_sample_frame_range_clamps_to_available_frames() -> None:
    metadata = VideoMetadata(path="x", fps=10.0, frame_count=80, width=1, height=1, duration_seconds=8.0)

    start_frame, end_frame = compute_sample_frame_range(
        metadata, start_seconds=0.0, duration_seconds=30.0, minimum_duration_seconds=None
    )

    assert (start_frame, end_frame) == (0, 80)


def test_compute_sample_frame_range_rejects_non_overlapping_start() -> None:
    metadata = VideoMetadata(path="x", fps=10.0, frame_count=50, width=1, height=1, duration_seconds=5.0)

    with pytest.raises(ValueError, match="does not overlap"):
        compute_sample_frame_range(
            metadata, start_seconds=10.0, duration_seconds=2.0, minimum_duration_seconds=None
        )


def test_read_frame_range_returns_frames_in_order(tmp_path: Path) -> None:
    video_path = _write_synthetic_video(tmp_path / "sample.avi", num_frames=20, fps=10.0)

    frames = read_frame_range(video_path, 5, 10)

    assert len(frames) == 5
    for offset, frame in enumerate(frames):
        expected_value = ((5 + offset) * 20) % 256
        assert frame.shape == (_HEIGHT, _WIDTH, 3)
        assert abs(int(frame.mean()) - expected_value) <= 5


def test_read_frame_range_rejects_invalid_bounds(tmp_path: Path) -> None:
    video_path = _write_synthetic_video(tmp_path / "sample.avi", num_frames=5, fps=10.0)
    with pytest.raises(ValueError, match="end_frame"):
        read_frame_range(video_path, 3, 3)


def test_render_optical_flow_video_hsv_mode_writes_readable_video(tmp_path: Path) -> None:
    frames = [_textured_frame(offset=i) for i in range(4)]
    output_path = tmp_path / "flow_hsv.mp4"

    result_path = render_optical_flow_video(frames, output_path, fps=10.0, mode="hsv")

    assert result_path.is_file()
    capture = cv2.VideoCapture(str(result_path))
    try:
        assert capture.isOpened()
        read_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (48, 48, 3)
            read_count += 1
    finally:
        capture.release()
    assert read_count == len(frames) - 1


def test_render_optical_flow_video_arrows_mode_writes_readable_video(tmp_path: Path) -> None:
    frames = [_textured_frame(offset=i) for i in range(3)]
    output_path = tmp_path / "flow_arrows.mp4"

    result_path = render_optical_flow_video(frames, output_path, fps=10.0, mode="arrows")

    capture = cv2.VideoCapture(str(result_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 1
    finally:
        capture.release()


def test_render_optical_flow_video_rejects_single_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two frames"):
        render_optical_flow_video([_textured_frame()], tmp_path / "out.mp4", fps=10.0)


def test_render_optical_flow_video_rejects_unknown_mode(tmp_path: Path) -> None:
    frames = [_textured_frame(offset=i) for i in range(2)]
    with pytest.raises(ValueError, match="unsupported mode"):
        render_optical_flow_video(frames, tmp_path / "out.mp4", fps=10.0, mode="rainbow")
