#!/usr/bin/env python3
"""Phase 6 real four-view planar Structure-From-Motion experiment.

Runs the actual assignment pipeline (IMPLEMENTATION_PLAN.md Section 14-20, Phase 6) on the
four real photographs the user captured of a single flat/planar object (the front cover of a
paperback book): ORB feature detection, descriptor matching, RANSAC homography estimation
(View 2/3/4 -> View 1), reprojection-error computation over RANSAC inliers only, and planar
boundary registration/reconstruction - and writes every real result to results/sfm/.

This is a real experiment script, not a fixture generator: every number it prints and saves
comes from actually processing data/sfm/view_1/IMG_7283.JPG through view_4/IMG_7286.JPG. It
raises rather than substituting a placeholder if a real input file is missing, so a partial or
mocked run can never be mistaken for a completed experiment.

Boundary corners
-----------------
The four real physical corners of the book's front cover were identified manually in each
view, following a reproducible two-step workflow (this script's ``_coarse_bbox`` reproduces
step 1 exactly; step 2 was visual):
  1. An HSV color threshold (the cover is a distinctive yellow/black print) plus
     ``cv2.connectedComponentsWithStats`` gives a coarse axis-aligned bounding box of the
     cover in each image - `_coarse_bbox` below, kept for reproducibility/inspection, not
     used directly as the corner coordinates (a rotated/oblique view's true corners are not
     the bounding box's own corners).
  2. Each of the four corners was then visually confirmed by inspecting a zoomed, pixel-grid
     overlaid crop of the image near that bounding-box edge (the same crop-and-zoom approach
     used for the Phase 4 manual tracking validation), and its pixel coordinate read directly
     off the grid.
Order in every array below: top-left, top-right, bottom-right, bottom-left (pixel coordinates,
origin top-left, x right, y down, on the EXIF-orientation-corrected image - see
``module5_6.io_utils.load_image_bgr_oriented``). These are real, manually observed
measurements, distinct from any homography-predicted location computed later in this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import ExifTags, Image, UnidentifiedImageError  # noqa: E402

from module5_6.camera import ViewMetadata  # noqa: E402
from module5_6.features import MatchParams, ORBParams, detect_and_describe, match_descriptors, matched_coordinates  # noqa: E402
from module5_6.geometry import apply_homography, boundary_polygon_closed  # noqa: E402
from module5_6.homography import HomographyParams  # noqa: E402
from module5_6.io_utils import load_image_bgr_oriented, to_grayscale  # noqa: E402
from module5_6.sfm import register_views  # noqa: E402
from module5_6.tracking import draw_tracked_points  # noqa: E402

DATA_DIR = _ROOT / "data" / "sfm"
RESULTS_DIR = _ROOT / "results" / "sfm"

VIEW_FILES = {
    "view_1": DATA_DIR / "view_1" / "IMG_7283.JPG",
    "view_2": DATA_DIR / "view_2" / "IMG_7284.JPG",
    "view_3": DATA_DIR / "view_3" / "IMG_7285.JPG",
    "view_4": DATA_DIR / "view_4" / "IMG_7286.JPG",
}
REFERENCE_VIEW_ID = "view_1"

# See module docstring for how these were obtained. (x, y) pixel coordinates, order
# top-left/top-right/bottom-right/bottom-left, on the EXIF-orientation-corrected image.
BOUNDARY_CORNERS_PX: dict[str, list[tuple[float, float]]] = {
    "view_1": [(785.0, 1730.0), (3055.0, 1698.0), (2995.0, 5305.0), (865.0, 5165.0)],
    "view_2": [(2450.0, 2135.0), (3260.0, 2195.0), (3085.0, 4410.0), (2400.0, 4440.0)],
    "view_3": [(1480.0, 1885.0), (2115.0, 1785.0), (2225.0, 4225.0), (1495.0, 3760.0)],
    "view_4": [(1440.0, 2280.0), (2870.0, 2400.0), (2665.0, 3760.0), (1600.0, 3560.0)],
}

# User-recorded approximate physical capture information (see IMPLEMENTATION_PLAN.md Phase 6
# kickoff instructions). Never treated as calibrated pose: no angle/rotation/translation is
# derived from these free-text descriptions anywhere in this script.
CAPTURE_NOTES: dict[str, dict[str, Any]] = {
    "view_1": {
        "orientation": "centered/front view, camera aimed approximately toward the object center",
        "distance_to_object_m": 0.2286,  # 9 inches, user-recorded approximate
        "distance_to_object_notes": "approximately 9 inches from the object (user-recorded, not measured with an instrument)",
        "role": "reference view (default; see docs/STRUCTURE_FROM_MOTION_THEORY.md for the choice rationale)",
    },
    "view_2": {
        "orientation": "left-side viewpoint, camera aimed back toward the object center",
        "distance_to_object_m": 0.6604,  # 26 inches, user-recorded approximate
        "distance_to_object_notes": "approximately 26 inches from the object (user-recorded, not measured with an instrument)",
        "role": "registered view",
    },
    "view_3": {
        "orientation": "right-side viewpoint, camera aimed back toward the object center",
        "distance_to_object_m": 0.6604,
        "distance_to_object_notes": "approximately 26 inches from the object (user-recorded, not measured with an instrument)",
        "role": "registered view",
    },
    "view_4": {
        "orientation": "vertically displaced / angled viewpoint, camera aimed toward the object",
        "distance_to_object_m": 0.6604,
        "distance_to_object_notes": "approximately 26 inches from the object (user-recorded, not measured with an instrument)",
        "role": "registered view",
    },
}

# Lowe's ratio test (0.75, the standard value from Lowe's SIFT paper, also widely used for
# ORB/BRIEF-family descriptors): keeps a match only when its best train-descriptor distance is
# below 0.75x the second-best distance. Diagnosis (see docs/EXPERIMENTAL_RESULTS.md and the
# Phase 6 report): an initial run using only a fixed Hamming-distance cutoff (no ratio test)
# registered View 3 with just 7/324 RANSAC inliers (2.2%), far below View 2/4's ~26-27%. The
# View 2/3/4 -> View 1 match visualizations showed correspondences concentrated on the book's
# printed title text and banner ("The Tempest", "UPDATED EDITION Folger SHAKESPEARE LIBRARY"),
# which contains repeated/self-similar letterforms; at View 3's viewing angle this produced
# many locally-plausible but globally-inconsistent matches between different occurrences of
# similar glyphs, which a plain best-distance cutoff cannot detect (the best distance alone can
# still look good) but the ratio test directly targets (an ambiguous match has a close second-
# best neighbor). This is a real, documented, reproducible parameter choice made in response to
# a diagnosed cause, per IMPLEMENTATION_PLAN.md Phase 6 Section 4 - not an arbitrary retry.
RATIO_TEST_THRESHOLD = 0.75
ORB_PARAMS = ORBParams(n_features=2000)
MATCH_PARAMS = MatchParams(ratio_test_threshold=RATIO_TEST_THRESHOLD)
HOMOGRAPHY_PARAMS = HomographyParams(method="ransac", ransac_reproj_threshold=3.0)

# A curated set of real EXIF tag names worth reporting; anything not present in a given file
# is simply omitted (never filled with a guessed value).
_EXIF_TAG_NAMES = {v: k for k, v in ExifTags.TAGS.items()}
_REPORTED_EXIF_FIELDS = [
    "Make",
    "Model",
    "LensModel",
    "FocalLength",
    "FocalLengthIn35mmFilm",
    "FNumber",
    "ExposureTime",
    "ISOSpeedRatings",
    "DateTime",
    "Orientation",
    "BrightnessValue",
    "ApertureValue",
    "ShutterSpeedValue",
]


def _exif_value_to_jsonable(value: Any) -> Any:
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, tuple):
        return [_exif_value_to_jsonable(v) for v in value]
    return value


_EXIF_SUB_IFD_TAG = 0x8769  # "Exif IFD Pointer": where FocalLength/FNumber/ExposureTime/etc live


def extract_real_exif(path: Path) -> dict[str, Any]:
    """Return only the real EXIF fields actually present in ``path``; never guesses a value.

    Checks both the base IFD (``Make``/``Model``/``Orientation``/``DateTime``) and the Exif
    sub-IFD (``FocalLength``, ``FNumber``, ``ExposureTime``, ``ISOSpeedRatings``, etc. - these
    live under the base IFD's ``ExifOffset``/0x8769 pointer, not the base IFD itself; a plain
    ``img.getexif()`` alone silently omits them).
    """
    try:
        with Image.open(path) as img:
            base = img.getexif()
            raw = dict(base)
            raw.update(base.get_ifd(_EXIF_SUB_IFD_TAG))
    except (OSError, UnidentifiedImageError):
        return {}
    found: dict[str, Any] = {}
    for field in _REPORTED_EXIF_FIELDS:
        tag_id = _EXIF_TAG_NAMES.get(field)
        if tag_id is not None and tag_id in raw:
            found[field] = _exif_value_to_jsonable(raw[tag_id])
    return found


def _coarse_bbox(image_bgr: np.ndarray) -> dict[str, int] | None:
    """Coarse axis-aligned bounding box of the yellow/black book cover via HSV thresholding.

    Reproducibility aid for the manual corner-identification workflow (see module docstring);
    not used as the boundary corner coordinates themselves.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (12, 50, 50), (42, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[largest]
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)}


def _save_jpg(path: Path, image_bgr: np.ndarray, *, quality: int = 90) -> None:
    """Save a report figure as JPEG (photographic content compresses far better than PNG here;
    these are visualization outputs, not pixel-critical data, so lossy compression is fine)."""
    cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _draw_boundary(image_bgr: np.ndarray, corners: np.ndarray, *, color: tuple[int, int, int], thickness: int = 6) -> np.ndarray:
    overlay = image_bgr.copy()
    closed = boundary_polygon_closed(corners).astype(int)
    for (x1, y1), (x2, y2) in zip(closed[:-1], closed[1:]):
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    for x, y in corners.astype(int):
        cv2.circle(overlay, (x, y), thickness + 4, color, -1)
    return overlay


def _draw_matches(
    view_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    points_view: np.ndarray,
    points_reference: np.ndarray,
    *,
    inlier_mask: np.ndarray | None = None,
    max_lines: int = 200,
) -> np.ndarray:
    """A side-by-side (view | reference) match visualization drawn with plain line segments.

    Not ``cv2.drawMatches`` because this pipeline works with plain ``(N, 2)`` arrays rather
    than ``cv2.KeyPoint``/``cv2.DMatch`` objects (see module5_6.features); downsamples the
    thumbnail scale so a full-resolution (5712x4284) photo pair stays a manageable file.
    """
    scale = 1600 / max(view_bgr.shape[0], view_bgr.shape[1])
    view_small = cv2.resize(view_bgr, None, fx=scale, fy=scale)
    reference_small = cv2.resize(reference_bgr, None, fx=scale, fy=scale)
    height = max(view_small.shape[0], reference_small.shape[0])
    canvas = np.zeros((height, view_small.shape[1] + reference_small.shape[1], 3), dtype=np.uint8)
    canvas[: view_small.shape[0], : view_small.shape[1]] = view_small
    canvas[: reference_small.shape[0], view_small.shape[1] :] = reference_small
    offset_x = view_small.shape[1]

    indices = np.arange(points_view.shape[0])
    if indices.size > max_lines:
        indices = np.linspace(0, indices.size - 1, max_lines).astype(int)
    for i in indices:
        is_inlier = inlier_mask is None or bool(inlier_mask[i])
        color = (0, 220, 0) if is_inlier else (0, 0, 220)
        p1 = (int(points_view[i, 0] * scale), int(points_view[i, 1] * scale))
        p2 = (int(points_reference[i, 0] * scale) + offset_x, int(points_reference[i, 1] * scale))
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.circle(canvas, p2, 3, color, -1)
    return canvas


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for view_id, path in VIEW_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"required real SfM input image missing: {path}")

    images_bgr: dict[str, np.ndarray] = {}
    grays: dict[str, np.ndarray] = {}
    views_summary: dict[str, Any] = {}

    print("== Loading real images (EXIF-orientation corrected) ==")
    for view_id, path in VIEW_FILES.items():
        image_bgr, orientation_used = load_image_bgr_oriented(path)
        images_bgr[view_id] = image_bgr
        grays[view_id] = to_grayscale(image_bgr)
        exif = extract_real_exif(path)
        bbox = _coarse_bbox(image_bgr)
        height, width = image_bgr.shape[:2]
        device = None
        if exif.get("Make") and exif.get("Model"):
            device = f"{exif['Make']} {exif['Model']}"
        metadata = ViewMetadata(
            view_id=view_id,
            image_path=f"data/sfm/{view_id}/{path.name}",
            width=width,
            height=height,
            device=device,
            focal_length_mm=exif.get("FocalLength"),
            distance_to_object_m=CAPTURE_NOTES[view_id]["distance_to_object_m"],
            orientation=CAPTURE_NOTES[view_id]["orientation"],
            notes=CAPTURE_NOTES[view_id]["distance_to_object_notes"],
            status="available",
        )
        views_summary[view_id] = {
            "file": str(path.relative_to(_ROOT)),
            "file_size_bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "exif_orientation_tag": orientation_used,
            "real_exif": exif,
            "view_metadata": metadata.to_dict(),
            "capture_notes_user_recorded": CAPTURE_NOTES[view_id],
            "boundary_corners_manual_px": BOUNDARY_CORNERS_PX[view_id],
            "coarse_bbox_hsv_autodetect": bbox,
        }
        print(f"  {view_id}: {path.name}  {width}x{height}  orientation={orientation_used}  device={device}")

    print("== ORB feature detection ==")
    keypoint_counts: dict[str, int] = {}
    for view_id, gray in grays.items():
        points, _ = detect_and_describe(gray, params=ORB_PARAMS)
        keypoint_counts[view_id] = int(points.shape[0])
        vis = draw_tracked_points(images_bgr[view_id], points, color=(0, 255, 0), radius=6)
        vis_small = cv2.resize(vis, None, fx=1600 / vis.shape[1], fy=1600 / vis.shape[1])
        _save_jpg(RESULTS_DIR / f"{view_id}_orb_keypoints.jpg", vis_small)
        views_summary[view_id]["orb_keypoints_artifact"] = f"results/sfm/{view_id}_orb_keypoints.jpg"
        print(f"  {view_id}: {points.shape[0]} ORB keypoints")

    print("== Matching, homography (RANSAC), reprojection, boundary registration ==")
    other_view_ids = [v for v in VIEW_FILES if v != REFERENCE_VIEW_ID]
    reference_boundary = np.array(BOUNDARY_CORNERS_PX[REFERENCE_VIEW_ID], dtype=np.float64)
    boundary_by_view = {v: np.array(BOUNDARY_CORNERS_PX[v], dtype=np.float64) for v in other_view_ids}

    sfm_result = register_views(
        REFERENCE_VIEW_ID,
        grays[REFERENCE_VIEW_ID],
        {v: grays[v] for v in other_view_ids},
        orb_params=ORB_PARAMS,
        match_params=MATCH_PARAMS,
        homography_params=HOMOGRAPHY_PARAMS,
        reference_boundary_points=reference_boundary,
        boundary_points_by_view=boundary_by_view,
    )

    registrations_summary: dict[str, Any] = {}
    math_workout: dict[str, Any] | None = None
    for view_id in other_view_ids:
        registration = sfm_result.registrations[view_id]

        # Candidate matches: unfiltered cross-check matches (before the Hamming-distance cutoff
        # applied via MATCH_PARAMS inside register_view/register_views above).
        view_points, view_descriptors = detect_and_describe(grays[view_id], params=ORB_PARAMS)
        ref_points, ref_descriptors = detect_and_describe(grays[REFERENCE_VIEW_ID], params=ORB_PARAMS)
        candidate_matches = match_descriptors(view_descriptors, ref_descriptors, params=MatchParams(cross_check=True))
        candidate_count = len(candidate_matches)
        retained_count = registration.matched_points_view.shape[0]
        inlier_mask = registration.inlier_mask
        inlier_count = int(inlier_mask.sum())
        inlier_errors = registration.reprojection_errors[inlier_mask]

        match_vis = _draw_matches(
            images_bgr[view_id], images_bgr[REFERENCE_VIEW_ID],
            registration.matched_points_view, registration.matched_points_reference,
            inlier_mask=None,
        )
        _save_jpg(RESULTS_DIR / f"{view_id}_to_{REFERENCE_VIEW_ID}_matches_all.jpg", match_vis)
        inlier_vis = _draw_matches(
            images_bgr[view_id], images_bgr[REFERENCE_VIEW_ID],
            registration.matched_points_view, registration.matched_points_reference,
            inlier_mask=inlier_mask,
        )
        _save_jpg(RESULTS_DIR / f"{view_id}_to_{REFERENCE_VIEW_ID}_matches_inliers.jpg", inlier_vis)

        height, width = images_bgr[REFERENCE_VIEW_ID].shape[:2]
        registered_warp = cv2.warpPerspective(images_bgr[view_id], registration.homography, (width, height))
        _save_jpg(
            RESULTS_DIR / f"{view_id}_registered_into_{REFERENCE_VIEW_ID}_frame.jpg",
            cv2.resize(registered_warp, None, fx=1600 / width, fy=1600 / width),
        )

        own_boundary_overlay = _draw_boundary(images_bgr[view_id], boundary_by_view[view_id], color=(0, 0, 255))
        _save_jpg(
            RESULTS_DIR / f"{view_id}_boundary_manual.jpg",
            cv2.resize(own_boundary_overlay, None, fx=1600 / width, fy=1600 / width),
        )

        registrations_summary[view_id] = {
            "orb_params": vars(ORB_PARAMS),
            "match_params": {
                "ratio_test_threshold": MATCH_PARAMS.ratio_test_threshold,
            },
            "homography_params": vars(HOMOGRAPHY_PARAMS),
            "reference_keypoint_count": keypoint_counts[REFERENCE_VIEW_ID],
            "view_keypoint_count": keypoint_counts[view_id],
            "candidate_match_count": candidate_count,
            "retained_match_count": retained_count,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_count / retained_count if retained_count else None,
            "homography_view_to_reference": registration.homography.tolist(),
            "reprojection_error_px": {
                "mean_inliers": float(np.mean(inlier_errors)) if inlier_errors.size else None,
                "median_inliers": float(np.median(inlier_errors)) if inlier_errors.size else None,
                "max_inliers": float(np.max(inlier_errors)) if inlier_errors.size else None,
                "count_inliers": int(inlier_errors.size),
            },
            "boundary_manual_view_px": boundary_by_view[view_id].tolist(),
            "boundary_registered_into_reference_px": registration.registered_boundary.tolist()
            if registration.registered_boundary is not None
            else None,
            "artifacts": {
                "orb_keypoints": f"results/sfm/{view_id}_orb_keypoints.jpg",
                "matches_all": f"results/sfm/{view_id}_to_{REFERENCE_VIEW_ID}_matches_all.jpg",
                "matches_inliers": f"results/sfm/{view_id}_to_{REFERENCE_VIEW_ID}_matches_inliers.jpg",
                "registered_warp": f"results/sfm/{view_id}_registered_into_{REFERENCE_VIEW_ID}_frame.jpg",
                "boundary_manual": f"results/sfm/{view_id}_boundary_manual.jpg",
            },
        }
        print(
            f"  {view_id} -> {REFERENCE_VIEW_ID}: candidates={candidate_count} retained={retained_count} "
            f"inliers={inlier_count} ({100 * inlier_count / retained_count:.1f}%)  "
            f"mean reproj err (inliers) = {registrations_summary[view_id]['reprojection_error_px']['mean_inliers']:.3f}px"
        )

        if math_workout is None and inlier_count > 0:
            inlier_indices = np.flatnonzero(inlier_mask)
            pick = int(inlier_indices[0])
            p_view = registration.matched_points_view[pick]
            p_ref_actual = registration.matched_points_reference[pick]
            H = registration.homography
            p_h = np.array([p_view[0], p_view[1], 1.0])
            q_h = H @ p_h
            predicted = np.array([q_h[0] / q_h[2], q_h[1] / q_h[2]])
            error = float(np.linalg.norm(predicted - p_ref_actual))
            math_workout = {
                "source_view": view_id,
                "reference_view": REFERENCE_VIEW_ID,
                "p_view_xy": p_view.tolist(),
                "homography_H": H.tolist(),
                "q_homogeneous_Hp": q_h.tolist(),
                "predicted_normalized_xy": predicted.tolist(),
                "actual_observed_reference_xy": p_ref_actual.tolist(),
                "reprojection_error_px": error,
            }

    print("== Boundary reconstruction (reference frame) ==")
    consensus = sfm_result.consensus_boundary
    reconstructed_overlay = _draw_boundary(images_bgr[REFERENCE_VIEW_ID], reference_boundary, color=(0, 0, 255))
    if consensus is not None:
        reconstructed_overlay = _draw_boundary(reconstructed_overlay, consensus, color=(0, 255, 255))
    height, width = images_bgr[REFERENCE_VIEW_ID].shape[:2]
    _save_jpg(
        RESULTS_DIR / "reference_boundary_reconstruction.jpg",
        cv2.resize(reconstructed_overlay, None, fx=1600 / width, fy=1600 / width),
    )
    own_boundary_overlay = _draw_boundary(images_bgr[REFERENCE_VIEW_ID], reference_boundary, color=(0, 0, 255))
    _save_jpg(
        RESULTS_DIR / "view_1_boundary_manual.jpg",
        cv2.resize(own_boundary_overlay, None, fx=1600 / width, fy=1600 / width),
    )

    # Optional normalized/top-down view: rectify View 1's own front cover onto an axis-aligned
    # rectangle sized from its manually observed boundary's real pixel edge lengths.
    top_width = int(round(max(
        np.linalg.norm(reference_boundary[1] - reference_boundary[0]),
        np.linalg.norm(reference_boundary[2] - reference_boundary[3]),
    )))
    top_height = int(round(max(
        np.linalg.norm(reference_boundary[3] - reference_boundary[0]),
        np.linalg.norm(reference_boundary[2] - reference_boundary[1]),
    )))
    target_rect = np.array([[0, 0], [top_width - 1, 0], [top_width - 1, top_height - 1], [0, top_height - 1]], dtype=np.float64)
    rectify_H, _ = cv2.findHomography(reference_boundary, target_rect, method=0)
    top_down = cv2.warpPerspective(images_bgr[REFERENCE_VIEW_ID], rectify_H, (top_width, top_height))
    _save_jpg(RESULTS_DIR / "view_1_top_down_rectified.jpg", top_down)

    summary = {
        "schema_version": 1,
        "object_description": (
            "Paperback book, front cover: 'The Tempest' by William Shakespeare, Folger "
            "Shakespeare Library Updated Edition (yellow/black textured cover). The front "
            "cover face is treated as the flat/2D planar object per the assignment's planar "
            "simplification; the spine and page edges visible in oblique views are not part "
            "of the tracked plane."
        ),
        "terminology_note": (
            "Planar Structure From Motion / projective (homography) registration experiment "
            "across four real viewpoints - not dense/full 3D reconstruction."
        ),
        "reference_view_id": REFERENCE_VIEW_ID,
        "reference_view_choice_rationale": (
            "View 1 (IMG_7283.JPG) is the centered/front viewpoint closest to the object "
            "(~9 in) with the least perspective foreshortening of the front cover face, per "
            "the user-recorded capture notes and visual inspection; used as the default "
            "reference per the assignment instructions."
        ),
        "views": views_summary,
        "registrations": registrations_summary,
        "boundary_reconstruction": {
            "reference_manual_boundary_px": reference_boundary.tolist(),
            "consensus_boundary_reference_frame_px": consensus.tolist() if consensus is not None else None,
            "consensus_method": (
                "Unweighted mean of View 1's own manually observed boundary and each of View "
                "2/3/4's manually observed boundary transformed into View 1's frame via that "
                "view's estimated homography (module5_6.sfm._consensus_boundary)."
            ),
            "artifacts": {
                "reference_boundary_reconstruction": "results/sfm/reference_boundary_reconstruction.jpg",
                "view_1_boundary_manual": "results/sfm/view_1_boundary_manual.jpg",
                "view_1_top_down_rectified": "results/sfm/view_1_top_down_rectified.jpg",
            },
        },
        "mathematical_workout_real_data": math_workout,
    }
    summary_path = RESULTS_DIR / "sfm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
