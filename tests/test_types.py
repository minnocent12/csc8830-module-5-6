from __future__ import annotations

from module5_6 import ExperimentStatus, ImageMetadata


def test_foundation_types_are_immutable_dataclasses() -> None:
    metadata = ImageMetadata(10, 20, 3, "uint8", "BGR", "frame_0001.png")
    status = ExperimentStatus("pending", "No video has been processed yet.")

    assert metadata.width == 20
    assert status.status == "pending"
