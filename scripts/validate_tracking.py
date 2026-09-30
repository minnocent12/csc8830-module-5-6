#!/usr/bin/env python3
"""Professor-required two-consecutive-frame manual pixel-location tracking validation.

Purpose
-------
IMPLEMENTATION_PLAN.md Section 12: for a chosen point P = (x1, y1) in Frame 1, compute the
Lucas-Kanade predicted location in Frame 2 (the "theoretical" result), separately and manually
record the actual observed corresponding location in Frame 2, and compute the Euclidean pixel
error between them:

    e = sqrt((predicted_x2 - observed_x)^2 + (predicted_y2 - observed_y)^2)

This is a reproducible two-step workflow (see also docs/TRACKING_VALIDATION.md):

  1. ``prepare``  reads two consecutive frames from a real video, detects/accepts a Frame 1
     point, computes the Lucas-Kanade prediction (module5_6.tracking, Phase 2), and writes
     Frame 1/Frame 2 evidence images plus a partial validation record
     (status: awaiting_manual_observation).
  2. ``record``   given that partial record and an observed Frame 2 coordinate that a human
     determined by inspecting the Frame 2 evidence image (e.g. in an image viewer that reports
     pixel coordinates under the cursor, or via the Motion Tracking web page's manual
     validation controls), computes the pixel error and writes the completed record plus a
     predicted-vs-observed overlay image.

Never pass the predicted coordinate back in as the observed coordinate for ``record`` - that
would not be a real observation and would defeat the purpose of this validation. Doing so
would also produce a pixel error of essentially zero that documents nothing.

Usage
-----
    python scripts/validate_tracking.py prepare \\
        --video path/to/video_1.mp4 --video-id video_1 --frame1 100 --point-label P1

    # ... visually inspect results/tracking/video_1/P1_frame2_predicted.png (or the original
    # Frame 2 image) to determine where the point actually is, then:

    python scripts/validate_tracking.py record \\
        --record results/tracking/video_1/P1_record.json \\
        --observed-x 412.5 --observed-y 208.0 \\
        --video path/to/video_1.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from module5_6.experiment import (  # noqa: E402
    TrackingValidationRecord,
    draw_validation_overlay,
    load_record,
    predict_frame2_location,
    record_observation,
    save_record,
)
from module5_6.io_utils import to_grayscale  # noqa: E402
from module5_6.tracking import ShiTomasiParams, detect_features, points_to_xy  # noqa: E402
from module5_6.video import read_frame_range  # noqa: E402


def _mark_point(frame_bgr: np.ndarray, x: float, y: float, color: tuple[int, int, int], label: str) -> np.ndarray:
    overlay = frame_bgr.copy()
    point = (int(round(x)), int(round(y)))
    cv2.drawMarker(overlay, point, color, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    cv2.putText(overlay, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return overlay


def _cmd_prepare(args: argparse.Namespace) -> Path:
    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")

    frames = read_frame_range(video_path, args.frame1, args.frame1 + 2)
    if len(frames) < 2:
        raise SystemExit("could not read two consecutive frames at the requested index")
    frame1, frame2 = frames[0], frames[1]
    gray1, gray2 = to_grayscale(frame1), to_grayscale(frame2)

    if args.x1 is not None and args.y1 is not None:
        x1, y1 = float(args.x1), float(args.y1)
    else:
        candidates = points_to_xy(detect_features(gray1, params=ShiTomasiParams(max_corners=20)))
        if candidates.shape[0] == 0:
            raise SystemExit("no Shi-Tomasi features found on Frame 1; supply --x1/--y1 explicitly")
        x1, y1 = float(candidates[0, 0]), float(candidates[0, 1])
        print(f"No --x1/--y1 given; auto-selected the strongest Shi-Tomasi corner at ({x1:.1f}, {y1:.1f}).")

    predicted_x2, predicted_y2, tracked_ok = predict_frame2_location(gray1, gray2, x1, y1)
    if not tracked_ok:
        print("WARNING: Lucas-Kanade reported this point as not reliably tracked (status=0).")

    record = TrackingValidationRecord(
        video_id=args.video_id,
        point_label=args.point_label,
        frame1_index=args.frame1,
        frame2_index=args.frame1 + 1,
        x1=x1,
        y1=y1,
        predicted_x2=predicted_x2,
        predicted_y2=predicted_y2,
    )

    output_dir = Path(args.output_dir) / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    frame1_path = output_dir / f"{args.point_label}_frame1.png"
    frame2_path = output_dir / f"{args.point_label}_frame2_predicted.png"
    record_path = output_dir / f"{args.point_label}_record.json"

    cv2.imwrite(str(frame1_path), _mark_point(frame1, x1, y1, (0, 255, 0), "P1"))
    cv2.imwrite(str(frame2_path), draw_validation_overlay(frame2, record))
    save_record(record, record_path)

    print(f"Prepared record: {record_path}")
    print(f"Frame 1 evidence: {frame1_path}")
    print(f"Frame 2 (predicted) evidence: {frame2_path}")
    print(
        "Next: visually inspect the Frame 2 evidence image (or the original Frame 2 frame) to "
        "determine the actual observed pixel location of this point, then run:\n"
        f"    python scripts/validate_tracking.py record --record {record_path} "
        "--observed-x X --observed-y Y --video " + str(video_path)
    )
    return record_path


def _cmd_record(args: argparse.Namespace) -> TrackingValidationRecord:
    record_path = Path(args.record)
    record = load_record(record_path)
    updated = record_observation(record, observed_x=args.observed_x, observed_y=args.observed_y, method=args.method)
    save_record(updated, record_path)

    print(f"Updated record: {record_path}")
    print(f"Pixel error e = {updated.pixel_error:.3f} px")

    if args.video:
        frame2 = read_frame_range(args.video, updated.frame2_index, updated.frame2_index + 1)[0]
        overlay_path = record_path.parent / f"{updated.point_label}_frame2_validated.png"
        cv2.imwrite(str(overlay_path), draw_validation_overlay(frame2, updated))
        print(f"Validated overlay: {overlay_path}")
    return updated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Extract frames and predict the Frame 2 location")
    prepare.add_argument("--video", required=True)
    prepare.add_argument("--video-id", required=True)
    prepare.add_argument("--frame1", type=int, required=True)
    prepare.add_argument("--point-label", default="P1")
    prepare.add_argument("--x1", type=float, default=None)
    prepare.add_argument("--y1", type=float, default=None)
    prepare.add_argument("--output-dir", default="results/tracking")
    prepare.set_defaults(func=_cmd_prepare)

    record = subparsers.add_parser("record", help="Record the manually observed Frame 2 location")
    record.add_argument("--record", required=True)
    record.add_argument("--observed-x", type=float, required=True)
    record.add_argument("--observed-y", type=float, required=True)
    record.add_argument("--method", default="manual pixel inspection")
    record.add_argument("--video", default=None, help="Original video, to regenerate the validated overlay")
    record.set_defaults(func=_cmd_record)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
