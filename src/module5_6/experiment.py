"""Phase 4 experimental tracking-validation infrastructure for Module 5-6.

Implements the professor-required manual pixel-location validation described in
IMPLEMENTATION_PLAN.md Section 12 and docs/TRACKING_VALIDATION.md: for a point ``P = (x1, y1)``
in Frame 1, compute the theoretical/algorithmic predicted location in Frame 2 using
pyramidal Lucas-Kanade tracking (``module5_6.tracking``, Phase 2), separately record the
*actual observed* location in Frame 2 (a real human measurement, never copied from the
prediction), and compute the Euclidean pixel error between them:

    e = sqrt((predicted_x2 - observed_x2)^2 + (predicted_y2 - observed_y2)^2)

This ``pixel_error`` is distinct from, and must never be confused with:

- OpenCV's own per-point Lucas-Kanade tracking error (``TrackedPoints.error``, Phase 2);
- forward-backward consistency error (``ForwardBackwardResult.fb_error``, Phase 2);
- any synthetic/known-translation error used only for automated software tests
  (``tests/test_tracking.py``, ``tests/test_optical_flow.py``).

A ``TrackingValidationRecord`` with ``observed_x``/``observed_y`` left as ``None`` is
explicitly ``PENDING_USER_EXPERIMENT`` and its ``pixel_error`` is ``None`` - no number is
reported until a real observation exists.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

from module5_6.tracking import LucasKanadeParams, track_points, xy_to_points

PENDING_USER_EXPERIMENT = "PENDING USER EXPERIMENT"

RecordStatus = Literal["awaiting_manual_observation", "complete"]


def predict_frame2_location(
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    x1: float,
    y1: float,
    *,
    params: LucasKanadeParams | None = None,
) -> tuple[float, float, bool]:
    """Compute the Lucas-Kanade predicted Frame 2 location for one Frame 1 point.

    This is the "theoretical/algorithmic predicted point location" step of
    IMPLEMENTATION_PLAN.md Section 12 - it reuses Phase 2's ``track_points`` for a single
    point. Returns ``(predicted_x2, predicted_y2, tracked_ok)``; ``tracked_ok`` is ``False``
    when OpenCV's own status output marks the track as failed, in which case the predicted
    coordinate should not be trusted for validation.
    """
    points = xy_to_points(np.array([[x1, y1]], dtype=np.float32))
    tracked = track_points(previous_gray, next_gray, points, params=params)
    return float(tracked.next_points[0, 0]), float(tracked.next_points[0, 1]), bool(tracked.status[0])


@dataclass
class TrackingValidationRecord:
    """One professor-required manual pixel-location validation record.

    Field names and the error formula follow IMPLEMENTATION_PLAN.md Section 12 exactly. This
    dataclass never computes ``observed_x``/``observed_y`` itself - those must be supplied by
    a real human observation via ``record_observation``.
    """

    video_id: str
    point_label: str
    frame1_index: int
    frame2_index: int
    x1: float
    y1: float
    predicted_x2: float
    predicted_y2: float
    observed_x: float | None = None
    observed_y: float | None = None
    status: RecordStatus = "awaiting_manual_observation"
    observation_method: str | None = None
    notes: str | None = None

    @property
    def predicted_displacement(self) -> tuple[float, float]:
        """``(u_hat, v_hat)``: predicted horizontal/vertical displacement, Frame 1 -> Frame 2."""
        return self.predicted_x2 - self.x1, self.predicted_y2 - self.y1

    @property
    def pixel_error(self) -> float | None:
        """Euclidean pixel error; ``None`` until a real observed coordinate is recorded."""
        if self.observed_x is None or self.observed_y is None:
            return None
        dx = self.predicted_x2 - self.observed_x
        dy = self.predicted_y2 - self.observed_y
        return float((dx * dx + dy * dy) ** 0.5)

    def to_dict(self) -> dict:
        data = asdict(self)
        u, v = self.predicted_displacement
        data["predicted_displacement_u"] = u
        data["predicted_displacement_v"] = v
        data["pixel_error"] = self.pixel_error
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrackingValidationRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def record_observation(
    record: TrackingValidationRecord, *, observed_x: float, observed_y: float, method: str
) -> TrackingValidationRecord:
    """Return a new record with the manually observed Frame 2 coordinate filled in.

    ``method`` must describe how the coordinate was actually obtained (e.g. "Streamlit Motion
    Tracking page - manual visual inspection") so the measurement stays reproducible. This
    function does not check that ``observed_x``/``observed_y`` differ from the prediction -
    that judgment belongs to the human doing the observation - but callers (the web page,
    ``scripts/validate_tracking.py``) should never pass the predicted coordinate straight back
    in as if it were observed.
    """
    return replace(
        record,
        observed_x=float(observed_x),
        observed_y=float(observed_y),
        status="complete",
        observation_method=method,
    )


def save_record(record: TrackingValidationRecord, path: str | Path) -> None:
    """Write one validation record as JSON."""
    record_path = Path(path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record.to_dict(), indent=2))


def load_record(path: str | Path) -> TrackingValidationRecord:
    """Load one validation record from JSON."""
    data = json.loads(Path(path).read_text())
    return TrackingValidationRecord.from_dict(data)


def records_to_table(records: Sequence[TrackingValidationRecord]) -> list[dict]:
    """Build report-ready rows matching IMPLEMENTATION_PLAN.md Section 12's required table."""
    rows = []
    for record in records:
        observed = (
            f"({record.observed_x:.2f}, {record.observed_y:.2f})"
            if record.observed_x is not None and record.observed_y is not None
            else "Pending"
        )
        error = f"{record.pixel_error:.3f}" if record.pixel_error is not None else "Pending"
        rows.append(
            {
                "Video": record.video_id,
                "Point": record.point_label,
                "Frame 1": f"({record.x1:.2f}, {record.y1:.2f}) @ frame {record.frame1_index}",
                "Predicted Frame 2": f"({record.predicted_x2:.2f}, {record.predicted_y2:.2f})",
                "Observed Frame 2": observed,
                "Pixel Error": error,
                "Status": record.status,
            }
        )
    return rows


_CSV_FIELDS = [
    "video_id",
    "point_label",
    "frame1_index",
    "frame2_index",
    "x1",
    "y1",
    "predicted_x2",
    "predicted_y2",
    "predicted_displacement_u",
    "predicted_displacement_v",
    "observed_x",
    "observed_y",
    "pixel_error",
    "status",
    "observation_method",
    "notes",
]


def write_records_csv(records: Sequence[TrackingValidationRecord], path: str | Path) -> None:
    """Write validation records to CSV, matching IMPLEMENTATION_PLAN.md Section 12's table."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            data = record.to_dict()
            writer.writerow({field: data.get(field) for field in _CSV_FIELDS})


def default_experiment_manifest() -> dict:
    """A fresh Phase 4 experiment manifest with both required videos marked pending.

    No video identity, path, or metadata is invented here - every field starts ``None`` or
    ``"pending_user_experiment"`` until a real video is supplied and processed.
    """
    return {
        "schema_version": 1,
        "videos": {
            "video_1": {
                "video_id": "video_1",
                "path": None,
                "status": "pending_user_experiment",
                "fps": None,
                "frame_count": None,
                "width": None,
                "height": None,
                "sample_start_seconds": None,
                "sample_duration_seconds": None,
                "optical_flow_video_path": None,
            },
            "video_2": {
                "video_id": "video_2",
                "path": None,
                "status": "pending_user_experiment",
                "fps": None,
                "frame_count": None,
                "width": None,
                "height": None,
                "sample_start_seconds": None,
                "sample_duration_seconds": None,
                "optical_flow_video_path": None,
            },
        },
        "validation_records": [],
    }


def load_experiment_manifest(path: str | Path) -> dict:
    """Load the experiment manifest, or return a fresh pending one if it does not exist yet."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return default_experiment_manifest()
    return json.loads(manifest_path.read_text())


def save_experiment_manifest(manifest: dict, path: str | Path) -> None:
    """Write the experiment manifest as JSON."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def find_supplied_video(directory: str | Path) -> Path | None:
    """Return the first real video file in a ``data/videos/<id>/`` directory, or ``None``.

    Ignores dotfiles such as ``.gitkeep`` so an empty, git-tracked placeholder directory is
    correctly treated as "no video supplied yet" rather than raising or guessing a fake path.
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return None
    for entry in sorted(directory_path.iterdir()):
        if entry.is_file() and not entry.name.startswith("."):
            return entry
    return None


def draw_validation_overlay(
    frame_bgr: np.ndarray,
    record: TrackingValidationRecord,
    *,
    predicted_color: tuple[int, int, int] = (0, 0, 255),
    observed_color: tuple[int, int, int] = (0, 255, 0),
    line_color: tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """Overlay the predicted (and, once recorded, observed) Frame 2 point with an error label.

    Never mutates ``frame_bgr``. When ``record.observed_x``/``observed_y`` are still ``None``
    (the pending case), only the predicted marker is drawn and no error is displayed.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a 3-channel BGR array")
    overlay = frame_bgr.copy()
    predicted_point = (int(round(record.predicted_x2)), int(round(record.predicted_y2)))
    cv2.drawMarker(overlay, predicted_point, predicted_color, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    cv2.putText(
        overlay, "predicted", (predicted_point[0] + 8, predicted_point[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, predicted_color, 1, cv2.LINE_AA,
    )
    if record.observed_x is not None and record.observed_y is not None:
        observed_point = (int(round(record.observed_x)), int(round(record.observed_y)))
        cv2.drawMarker(
            overlay, observed_point, observed_color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2
        )
        cv2.putText(
            overlay, "observed", (observed_point[0] + 8, observed_point[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, observed_color, 1, cv2.LINE_AA,
        )
        cv2.line(overlay, predicted_point, observed_point, line_color, 1, cv2.LINE_AA)
        error = record.pixel_error
        if error is not None:
            cv2.putText(
                overlay, f"error = {error:.2f} px", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
    return overlay
