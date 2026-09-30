from __future__ import annotations

import pytest

from module5_6.camera import ViewMetadata, intrinsic_matrix_from_values


def test_view_metadata_defaults_are_none_and_pending() -> None:
    view = ViewMetadata(view_id="view_1")

    assert view.image_path is None
    assert view.width is None
    assert view.height is None
    assert view.device is None
    assert view.focal_length_mm is None
    assert view.intrinsic_matrix is None
    assert view.position_xyz is None
    assert view.orientation is None
    assert view.distance_to_object_m is None
    assert view.zoom_changed is None
    assert view.notes is None
    assert view.status == "pending_user_experiment"


def test_view_metadata_no_fabricated_defaults_for_physical_parameters() -> None:
    """Constructing with only an id and real image dimensions must not invent optics."""
    view = ViewMetadata(view_id="view_1", width=4032, height=3024)

    # Real, measured dimensions are allowed; nothing derived from them is fabricated.
    assert view.width == 4032
    assert view.height == 3024
    assert view.focal_length_mm is None
    assert view.intrinsic_matrix is None
    assert view.position_xyz is None


def test_view_metadata_preserves_supplied_values() -> None:
    view = ViewMetadata(
        view_id="view_2",
        image_path="data/sfm/view_2/photo.jpg",
        width=4032,
        height=3024,
        device="iPhone 13, rear camera",
        focal_length_mm=26.0,
        intrinsic_matrix=[[3000.0, 0.0, 2016.0], [0.0, 3000.0, 1512.0], [0.0, 0.0, 1.0]],
        position_xyz=(0.5, 0.0, 1.2),
        orientation="facing object, camera held level",
        distance_to_object_m=1.0,
        zoom_changed=False,
        notes="tripod-mounted",
        status="available",
    )

    assert view.device == "iPhone 13, rear camera"
    assert view.focal_length_mm == 26.0
    assert view.intrinsic_matrix == [[3000.0, 0.0, 2016.0], [0.0, 3000.0, 1512.0], [0.0, 0.0, 1.0]]
    assert view.position_xyz == (0.5, 0.0, 1.2)
    assert view.status == "available"


def test_view_metadata_to_dict_and_from_dict_roundtrip() -> None:
    view = ViewMetadata(
        view_id="view_3",
        width=1920,
        height=1080,
        position_xyz=(1.0, 2.0, 3.0),
        status="available",
    )

    data = view.to_dict()
    restored = ViewMetadata.from_dict(data)

    assert restored == view


def test_view_metadata_from_dict_ignores_unknown_keys() -> None:
    restored = ViewMetadata.from_dict({"view_id": "view_4", "unexpected_field": 123})

    assert restored.view_id == "view_4"
    assert restored.width is None


def test_view_metadata_rejects_empty_view_id() -> None:
    with pytest.raises(ValueError, match="view_id"):
        ViewMetadata(view_id="")


def test_view_metadata_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="width"):
        ViewMetadata(view_id="view_1", width=-1)
    with pytest.raises(ValueError, match="height"):
        ViewMetadata(view_id="view_1", height=0)


def test_view_metadata_rejects_non_positive_focal_length_and_distance() -> None:
    with pytest.raises(ValueError, match="focal_length_mm"):
        ViewMetadata(view_id="view_1", focal_length_mm=0.0)
    with pytest.raises(ValueError, match="distance_to_object_m"):
        ViewMetadata(view_id="view_1", distance_to_object_m=-2.0)


def test_view_metadata_rejects_malformed_intrinsic_matrix() -> None:
    with pytest.raises(ValueError, match="3x3"):
        ViewMetadata(view_id="view_1", intrinsic_matrix=[[1.0, 0.0], [0.0, 1.0]])


def test_intrinsic_matrix_from_values_builds_expected_structure() -> None:
    K = intrinsic_matrix_from_values(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)

    assert K == [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]
