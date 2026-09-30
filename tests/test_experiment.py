from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from module5_6.experiment import (
    TrackingValidationRecord,
    default_experiment_manifest,
    draw_validation_overlay,
    find_supplied_video,
    load_experiment_manifest,
    load_record,
    predict_frame2_location,
    record_observation,
    save_experiment_manifest,
    save_record,
    write_records_csv,
)


def _textured_frame(size: tuple[int, int] = (160, 160), *, offset: int = 0) -> np.ndarray:
    """A deterministic textured grayscale frame; software verification only, not assignment data."""
    rng = np.random.default_rng(offset)
    return rng.integers(0, 256, size=size, dtype=np.uint8)


def _translated(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    height, width = frame.shape[:2]
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def _sample_record(**overrides) -> TrackingValidationRecord:
    defaults = dict(
        video_id="video_1",
        point_label="P1",
        frame1_index=10,
        frame2_index=11,
        x1=50.0,
        y1=60.0,
        predicted_x2=55.0,
        predicted_y2=63.0,
    )
    defaults.update(overrides)
    return TrackingValidationRecord(**defaults)


def test_predict_frame2_location_recovers_known_synthetic_translation() -> None:
    """A known synthetic (dx, dy) shift is software verification, not assignment evidence."""
    dx, dy = 4, -2
    previous = _textured_frame()
    following = _translated(previous, dx, dy)

    predicted_x2, predicted_y2, tracked_ok = predict_frame2_location(previous, following, 80.0, 80.0)

    assert tracked_ok is True
    assert predicted_x2 == pytest.approx(80.0 + dx, abs=1.0)
    assert predicted_y2 == pytest.approx(80.0 + dy, abs=1.0)


def test_pixel_error_is_none_until_observation_is_recorded() -> None:
    record = _sample_record()
    assert record.status == "awaiting_manual_observation"
    assert record.pixel_error is None


def test_pixel_error_matches_the_assignment_formula() -> None:
    """e = sqrt((predicted_x2 - observed_x)^2 + (predicted_y2 - observed_y)^2), Section 12."""
    record = _sample_record(predicted_x2=55.0, predicted_y2=63.0)

    updated = record_observation(record, observed_x=52.0, observed_y=60.0, method="test")

    expected = ((55.0 - 52.0) ** 2 + (63.0 - 60.0) ** 2) ** 0.5
    assert updated.pixel_error == pytest.approx(expected)
    assert updated.status == "complete"
    assert updated.observation_method == "test"


def test_record_observation_does_not_mutate_the_original_record() -> None:
    record = _sample_record()
    record_observation(record, observed_x=1.0, observed_y=1.0, method="test")

    assert record.observed_x is None
    assert record.status == "awaiting_manual_observation"


def test_predicted_displacement_matches_manual_subtraction() -> None:
    record = _sample_record(x1=10.0, y1=20.0, predicted_x2=13.0, predicted_y2=17.0)

    u, v = record.predicted_displacement

    assert (u, v) == pytest.approx((3.0, -3.0))


def test_save_and_load_record_roundtrip(tmp_path: Path) -> None:
    record = record_observation(_sample_record(), observed_x=54.0, observed_y=61.5, method="manual")
    path = tmp_path / "record.json"

    save_record(record, path)
    loaded = load_record(path)

    assert loaded.video_id == record.video_id
    assert loaded.observed_x == pytest.approx(54.0)
    assert loaded.pixel_error == pytest.approx(record.pixel_error)
    assert loaded.status == "complete"


def test_record_to_dict_includes_computed_fields() -> None:
    record = record_observation(_sample_record(), observed_x=54.0, observed_y=61.5, method="manual")

    data = record.to_dict()

    assert "pixel_error" in data
    assert "predicted_displacement_u" in data
    assert "predicted_displacement_v" in data
    assert data["pixel_error"] == pytest.approx(record.pixel_error)


def test_write_records_csv_contains_expected_rows(tmp_path: Path) -> None:
    pending = _sample_record(point_label="P1")
    complete = record_observation(_sample_record(point_label="P2"), observed_x=54.0, observed_y=61.5, method="manual")
    path = tmp_path / "records.csv"

    write_records_csv([pending, complete], path)

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["point_label"] == "P1"
    assert rows[0]["pixel_error"] == ""
    assert rows[1]["point_label"] == "P2"
    assert float(rows[1]["pixel_error"]) == pytest.approx(complete.pixel_error)


def test_default_experiment_manifest_marks_both_videos_pending() -> None:
    manifest = default_experiment_manifest()

    assert manifest["videos"]["video_1"]["status"] == "pending_user_experiment"
    assert manifest["videos"]["video_2"]["status"] == "pending_user_experiment"
    assert manifest["videos"]["video_1"]["path"] is None
    assert manifest["validation_records"] == []


def test_load_experiment_manifest_returns_default_when_file_is_missing(tmp_path: Path) -> None:
    manifest = load_experiment_manifest(tmp_path / "does_not_exist.json")

    assert manifest == default_experiment_manifest()


def test_save_and_load_experiment_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = default_experiment_manifest()
    manifest["videos"]["video_1"]["status"] = "available"
    manifest["videos"]["video_1"]["path"] = "data/videos/video_1/example.mp4"
    path = tmp_path / "manifest.json"

    save_experiment_manifest(manifest, path)
    loaded = load_experiment_manifest(path)

    assert loaded["videos"]["video_1"]["status"] == "available"
    assert loaded["videos"]["video_1"]["path"] == "data/videos/video_1/example.mp4"
    assert loaded["videos"]["video_2"]["status"] == "pending_user_experiment"


def test_find_supplied_video_ignores_gitkeep_placeholder(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").touch()

    assert find_supplied_video(tmp_path) is None


def test_find_supplied_video_finds_a_real_file(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").touch()
    video_file = tmp_path / "my_video.mp4"
    video_file.write_bytes(b"not a real video, just a placeholder for the file-discovery test")

    found = find_supplied_video(tmp_path)

    assert found == video_file


def test_find_supplied_video_returns_none_for_missing_directory(tmp_path: Path) -> None:
    assert find_supplied_video(tmp_path / "nonexistent") is None


def test_draw_validation_overlay_pending_case_does_not_mutate_and_has_no_error_text() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    original = frame.copy()
    record = _sample_record()

    overlay = draw_validation_overlay(frame, record)

    np.testing.assert_array_equal(frame, original)
    assert overlay.shape == frame.shape
    assert not np.array_equal(overlay, frame)


def test_draw_validation_overlay_complete_case_differs_from_pending_case() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    pending = _sample_record()
    complete = record_observation(pending, observed_x=58.0, observed_y=66.0, method="manual")

    pending_overlay = draw_validation_overlay(frame, pending)
    complete_overlay = draw_validation_overlay(frame, complete)

    assert not np.array_equal(pending_overlay, complete_overlay)


def test_draw_validation_overlay_rejects_non_bgr_frame() -> None:
    with pytest.raises(ValueError, match="3-channel BGR"):
        draw_validation_overlay(np.zeros((10, 10), dtype=np.uint8), _sample_record())
