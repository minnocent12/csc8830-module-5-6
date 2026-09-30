"""Camera/view metadata for Module 5-6 planar Structure From Motion.

Implements the per-view record IMPLEMENTATION_PLAN.md Section 14 asks for. No physical camera
parameter is ever invented or inferred from image size: focal length, intrinsic matrix,
position, orientation, and distance-to-object all default to ``None`` and stay ``None`` until
a real value is actually supplied. See IMPLEMENTATION_PLAN.md Section 15 and
docs/CAMERA_GEOMETRY.md for the pinhole camera model (``K``, ``[R|t]``) these fields describe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Literal

ViewStatus = Literal["pending_user_experiment", "available"]


@dataclass
class ViewMetadata:
    """Real, measured/recorded information about one of the four assignment viewpoints.

    Every physical-measurement field is optional and defaults to ``None``: constructing a
    ``ViewMetadata`` with only a ``view_id`` describes an unsupplied view honestly, rather than
    filling in a plausible-looking default. ``status`` tracks whether a real image/measurement
    for this view has actually been supplied yet.
    """

    view_id: str
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    device: str | None = None
    focal_length_mm: float | None = None
    intrinsic_matrix: list[list[float]] | None = None  # 3x3 K, only if actually known
    position_xyz: tuple[float, float, float] | None = None  # approximate, only if supplied
    orientation: str | None = None  # free-form description, only if supplied
    distance_to_object_m: float | None = None
    zoom_changed: bool | None = None
    notes: str | None = None
    status: ViewStatus = "pending_user_experiment"

    def __post_init__(self) -> None:
        if not self.view_id:
            raise ValueError("view_id must be a non-empty string")
        if self.width is not None and self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height is not None and self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.focal_length_mm is not None and self.focal_length_mm <= 0:
            raise ValueError(f"focal_length_mm must be positive, got {self.focal_length_mm}")
        if self.distance_to_object_m is not None and self.distance_to_object_m <= 0:
            raise ValueError(f"distance_to_object_m must be positive, got {self.distance_to_object_m}")
        if self.intrinsic_matrix is not None:
            rows = len(self.intrinsic_matrix)
            if rows != 3 or any(len(row) != 3 for row in self.intrinsic_matrix):
                raise ValueError("intrinsic_matrix must be a 3x3 nested list when supplied")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ViewMetadata":
        known = {f.name for f in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in known}
        if "position_xyz" in filtered and filtered["position_xyz"] is not None:
            filtered["position_xyz"] = tuple(filtered["position_xyz"])
        return cls(**filtered)


def intrinsic_matrix_from_values(fx: float, fy: float, cx: float, cy: float) -> list[list[float]]:
    """Build the pinhole intrinsic matrix K from explicitly supplied focal/principal-point values.

    Never call this with values inferred from image dimensions alone - `fx`/`fy`/`cx`/`cy`
    must come from an actual calibration or a device's reported/measured optics.
    """
    return [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]
