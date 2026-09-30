#!/usr/bin/env python3
"""Compute and save optical-flow evidence for one Module 5-6 assignment video.

Purpose
-------
Reads a real, user-supplied video (never a synthetic test fixture), extracts a sample
interval (defaulting to the assignment's required 30-second minimum, per
IMPLEMENTATION_PLAN.md Section 4), computes dense Farneback optical flow between consecutive
frames (``module5_6.optical_flow``, Phase 1), writes a visualization video, and records
magnitude/direction summary statistics as JSON. This produces Question 1's required "optical
flow computed and visualized as a video" evidence for one video.

This script does NOT by itself satisfy the two-consecutive-frame manual pixel-location
validation required by IMPLEMENTATION_PLAN.md Section 12 - use ``validate_tracking.py`` for
that.

Usage
-----
    python scripts/process_optical_flow.py --video path/to/video_1.mp4 --video-id video_1

Optional arguments:
    --start-seconds FLOAT       Sample start time (default 0.0)
    --duration-seconds FLOAT    Sample duration (default 30.0, the assignment minimum)
    --mode {hsv,arrows}         Visualization mode (default hsv)
    --output-dir PATH           Base output directory (default results/optical_flow)
    --allow-short-sample        Allow a sample shorter than 30 seconds (NOT valid for the
                                 assignment submission; for quick local previews only)

Outputs (under <output-dir>/<video-id>/):
    <video-id>_optical_flow_<mode>.mp4    the optical-flow visualization video
    <video-id>_optical_flow_summary.json  sample metadata and magnitude statistics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from module5_6.io_utils import to_grayscale  # noqa: E402
from module5_6.optical_flow import (  # noqa: E402
    compute_farneback_flow,
    flow_magnitude_angle,
    summarize_flow_magnitudes,
)
from module5_6.video import (  # noqa: E402
    MINIMUM_SAMPLE_DURATION_SECONDS,
    compute_sample_frame_range,
    get_video_metadata,
    read_frame_range,
    render_optical_flow_video,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to a real assignment video")
    parser.add_argument("--video-id", required=True, help='e.g. "video_1" or "video_2"')
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=MINIMUM_SAMPLE_DURATION_SECONDS)
    parser.add_argument("--mode", choices=["hsv", "arrows"], default="hsv")
    parser.add_argument("--output-dir", default="results/optical_flow")
    parser.add_argument(
        "--allow-short-sample",
        action="store_true",
        help="Allow a sample shorter than the assignment's 30-second minimum (preview only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")

    metadata = get_video_metadata(video_path)
    minimum = None if args.allow_short_sample else MINIMUM_SAMPLE_DURATION_SECONDS
    start_frame, end_frame = compute_sample_frame_range(
        metadata,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        minimum_duration_seconds=minimum,
    )
    frames = read_frame_range(video_path, start_frame, end_frame)
    grays = [to_grayscale(frame) for frame in frames]
    flows = [compute_farneback_flow(grays[i], grays[i + 1]) for i in range(len(grays) - 1)]
    magnitudes = [flow_magnitude_angle(flow)[0] for flow in flows]
    stats = summarize_flow_magnitudes(magnitudes)

    output_dir = Path(args.output_dir) / args.video_id
    output_video_path = output_dir / f"{args.video_id}_optical_flow_{args.mode}.mp4"
    render_optical_flow_video(frames, output_video_path, fps=metadata.fps, mode=args.mode)

    summary = {
        "video_id": args.video_id,
        "source_video": str(video_path),
        "fps": metadata.fps,
        "frame_count": metadata.frame_count,
        "width": metadata.width,
        "height": metadata.height,
        "sample_start_frame": start_frame,
        "sample_end_frame": end_frame,
        "sample_frame_count": len(frames),
        "sample_duration_seconds": len(frames) / metadata.fps,
        "mode": args.mode,
        "output_video": str(output_video_path),
        "mean_magnitude": stats.mean_magnitude,
        "median_magnitude": stats.median_magnitude,
        "max_magnitude": stats.max_magnitude,
        "frame_pairs": stats.frame_pairs,
    }
    summary_path = output_dir / f"{args.video_id}_optical_flow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote optical-flow video: {output_video_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
