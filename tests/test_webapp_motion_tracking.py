"""Motion Tracking page: structure of the redesigned page and its unchanged functional contract.

The live-path tests use a tiny synthetic translating-texture clip written to a temporary
folder. It is software verification only, never assignment evidence.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
INTRO_START = "Question 1: Shi-Tomasi feature detection plus pyramidal Lucas-Kanade tracking"

# Every configuration widget the page owns: key -> (type, label, default).
CONFIG_WIDGETS = {
    "tracking_start_seconds": ("number_input", "Start time (seconds)", 0.0),
    "tracking_frames_to_load": ("slider", "Frames to load (track-history length)", 15),
    "tracking_max_corners": ("slider", "Max corners", 100),
    "tracking_quality_level": ("slider", "Quality level", 0.3),
    "tracking_min_distance": ("slider", "Min distance between corners (pixels)", 7.0),
    "tracking_block_size": ("select_slider", "Block size", 7),
    "tracking_use_harris": ("checkbox", "Use Harris corner detector", False),
    "tracking_win_size": ("select_slider", "Window size", 21),
    "tracking_max_level": ("slider", "Pyramid levels", 3),
    "tracking_max_iterations": ("slider", "Max iterations", 30),
    "tracking_epsilon": ("slider", "Convergence epsilon", 0.01),
    "tracking_validate_fb": ("checkbox", "Validate tracks with forward-backward error", True),
    "tracking_max_fb_error": ("slider", "Max forward-backward error (pixels)", 1.0),
}
VALIDATION_WIDGETS = {
    "tracking_validation_point_index": ("number_input", "Point index to validate (row number from the table above)", 0),
    "tracking_validation_video_id": ("text_input", "Video ID for this record", "video_1"),
    "tracking_validation_point_label": ("text_input", "Point label", "P1"),
    "tracking_validation_confirmed": (
        "checkbox",
        "I have visually inspected Frame 2 above and the coordinates below reflect what I "
        "actually observed (not copied from the prediction)",
        False,
    ),
}
LIVE_METRICS = ["Detected features", "Valid tracks", "Mean |displacement| (px)", "Max |displacement| (px)"]

needs_upload_api = pytest.mark.skipif(
    not hasattr(AppTest, "file_uploader"), reason="AppTest.file_uploader needs streamlit>=1.56"
)


def _open() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.sidebar.radio[0].set_value("Motion Tracking").run()
    assert not app.exception
    return app


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory) -> tuple[str, bytes, str]:
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur((rng.random((120, 160)) * 255).astype(np.uint8), (5, 5), 0)
    path = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for i in range(12):
        writer.write(cv2.cvtColor(np.roll(texture, shift=(i, i), axis=(1, 0)), cv2.COLOR_GRAY2BGR))
    writer.release()
    return ("synthetic.mp4", path.read_bytes(), "video/mp4")


def _uploaded(clip) -> AppTest:
    app = _open()
    app.file_uploader(key="tracking_upload").set_value(clip).run()
    assert not app.exception
    return app


# Header and bundled evidence (no upload)


def test_page_header_renders_the_title_once_with_the_assignment_intro() -> None:
    app = _open()
    assert [h.value for h in app.header] == ["Motion Tracking"]
    assert any(m.value.startswith(INTRO_START) for m in app.markdown)
    assert not any(i.value.startswith("Question 1") for i in app.info)  # no longer an alert
    assert any('class="csc8830-eyebrow">Module 5-6<' in e.proto.body for e in app.get("html"))


def test_bundled_evidence_keeps_its_provenance_and_exact_metrics() -> None:
    app = _open()
    assert any(i.value.startswith("No video uploaded, so this shows the completed real") for i in app.info)
    assert "Committed Tracking Evidence" in [s.value for s in app.subheader]
    assert [(m.label, m.value) for m in app.metric] == [
        ("Predicted Frame 2", "(226.3, 1705.4)"),
        ("Observed Frame 2 (manual)", "(225.0, 1701.0)"),
        ("Pixel error e (px)", "4.628"),
        ("Predicted Frame 2", "(1115.6, 1950.7)"),
        ("Observed Frame 2 (manual)", "(1110.0, 1957.0)"),
        ("Pixel error e (px)", "8.408"),
    ]


def test_native_uploader_is_present_with_its_key_and_label() -> None:
    app = _open()
    uploader = next(e for e in app.get("file_uploader"))
    assert uploader.proto.label == "Video"
    assert "tracking_upload" in uploader.proto.id


# Live path


@needs_upload_api
def test_configuration_widgets_keep_keys_labels_and_defaults(synthetic_clip) -> None:
    app = _uploaded(synthetic_clip)
    for key, (kind, label, default) in CONFIG_WIDGETS.items():
        widget = getattr(app, kind)(key=key)
        assert widget.label == label, key
        assert widget.value == pytest.approx(default), key


@needs_upload_api
def test_run_gate_is_explicit_and_the_button_is_primary(synthetic_clip) -> None:
    app = _uploaded(synthetic_clip)
    button = app.button[0]
    assert button.label == "Detect and track features"
    assert button.proto.type == "primary"
    assert not app.metric and not app.dataframe  # nothing runs before the click
    assert any("select Detect and track features to run the experiment" in i.value for i in app.info)
    assert not any("PENDING" in w.value or "foundation phase" in w.value for w in app.warning)


@needs_upload_api
def test_run_produces_native_metrics_table_and_validation_widgets(synthetic_clip) -> None:
    app = _uploaded(synthetic_clip)
    app.button[0].click().run()
    assert not app.exception
    assert [m.label for m in app.metric] == LIVE_METRICS
    assert len(app.dataframe) == 1
    assert list(app.dataframe[0].value.columns)[:3] == ["Point", "Frame1 x", "Frame1 y"]
    for key, (kind, label, default) in VALIDATION_WIDGETS.items():
        widget = getattr(app, kind)(key=key)
        assert (widget.label, widget.value) == (label, default), key
    subheaders = [s.value for s in app.subheader]
    for section in ("Tracking Evidence", "Selected Statistics", "Tracking Data", "Manual Validation", "Interpretation"):
        assert section in subheaders


@needs_upload_api
def test_confirmed_validation_shows_pixel_error_and_native_download(synthetic_clip) -> None:
    app = _uploaded(synthetic_clip)
    app.button[0].click().run()
    app.checkbox(key="tracking_validation_confirmed").check().run()  # a normal later rerun
    assert not app.exception
    assert "Pixel error e (predicted vs. observed)" in [m.label for m in app.metric]
    downloads = app.get("download_button")
    assert [d.proto.label for d in downloads] == ["Download validation record (JSON)"]


def test_downloaded_record_shape_is_unchanged() -> None:
    from module5_6.experiment import TrackingValidationRecord, record_observation

    record = record_observation(
        TrackingValidationRecord(
            video_id="video_1", point_label="P1", frame1_index=0, frame2_index=1,
            x1=1.0, y1=2.0, predicted_x2=3.0, predicted_y2=4.0,
        ),
        observed_x=3.0,
        observed_y=5.0,
        method="Streamlit Motion Tracking page - manual visual inspection",
    )
    assert list(record.to_dict()) == [
        "video_id", "point_label", "frame1_index", "frame2_index", "x1", "y1",
        "predicted_x2", "predicted_y2", "observed_x", "observed_y", "status",
        "observation_method", "notes", "predicted_displacement_u",
        "predicted_displacement_v", "pixel_error",
    ]


# Completed runs survive the reruns caused by manual validation


@needs_upload_api
def test_real_user_flow_completes_manual_validation_across_reruns(synthetic_clip) -> None:
    """Regression: ticking the confirmation used to rerun with the button released and hide everything."""
    app = _uploaded(synthetic_clip)
    app.button[0].click().run()
    assert "tracking_completed_run" in app.session_state
    app.number_input(key="tracking_validation_point_index").set_value(1).run()  # button now False
    assert [m.label for m in app.metric] == LIVE_METRICS  # the stored result is still shown
    assert len(app.dataframe) == 1
    app.checkbox(key="tracking_validation_confirmed").check().run()
    app.number_input(key="tracking_validation_observed_x").set_value(12.0).run()
    assert not app.exception
    assert app.metric[-1].label == "Pixel error e (predicted vs. observed)"
    assert app.metric[-1].value.endswith(" px")
    assert [d.proto.label for d in app.get("download_button")] == ["Download validation record (JSON)"]
    assert not any("pending until you inspect" in w.value for w in app.warning)


@needs_upload_api
@pytest.mark.parametrize(
    "kind, key, value",
    [
        ("slider", "tracking_frames_to_load", 10),  # frame settings
        ("slider", "tracking_max_corners", 50),  # Shi-Tomasi
        ("select_slider", "tracking_win_size", 31),  # Lucas-Kanade
        ("checkbox", "tracking_validate_fb", False),  # forward-backward validation
    ],
)
def test_changing_a_computation_setting_invalidates_the_stored_result(synthetic_clip, kind, key, value) -> None:
    app = _uploaded(synthetic_clip)
    app.button[0].click().run()
    assert app.metric
    getattr(app, kind)(key=key).set_value(value).run()
    assert not app.exception
    assert not app.metric and not app.dataframe and not app.get("download_button")
    assert any("settings changed since the last run" in i.value for i in app.info)
    assert "tracking_completed_run" not in app.session_state
    app.button[0].click().run()  # a new explicit run is required, and works
    assert [m.label for m in app.metric] == LIVE_METRICS


@needs_upload_api
def test_a_new_upload_never_reuses_the_previous_result(synthetic_clip) -> None:
    app = _uploaded(synthetic_clip)
    app.button[0].click().run()
    name, data, mime = synthetic_clip
    app.file_uploader(key="tracking_upload").set_value((name, data, mime)).run()  # same name again
    assert not app.metric
    assert any("settings changed since the last run" in i.value for i in app.info)


@needs_upload_api
def test_validation_reruns_never_recompute_tracking(synthetic_clip) -> None:
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(APP_PATH).parent / 'src')!r})\n"
        "import streamlit as st\n"
        "from module5_6.webapp import pages\n"
        "if not hasattr(pages, '_real_detect_features'):  # the module persists across reruns\n"
        "    pages._real_detect_features = pages.detect_features\n"
        "    def counting(*args, **kwargs):\n"
        "        st.session_state['detect_calls'] = st.session_state.get('detect_calls', 0) + 1\n"
        "        return pages._real_detect_features(*args, **kwargs)\n"
        "    pages.detect_features = counting\n"
        "pages._motion_tracking_page()\n"
    )
    from module5_6.webapp import pages

    try:
        app = AppTest.from_string(script, default_timeout=60).run()
        app.file_uploader(key="tracking_upload").set_value(synthetic_clip).run()
        app.button[0].click().run()
        assert app.session_state["detect_calls"] == 1
        app.checkbox(key="tracking_validation_confirmed").check().run()
        app.number_input(key="tracking_validation_observed_y").set_value(9.0).run()
        app.text_input(key="tracking_validation_point_label").set_value("P2").run()
        assert app.session_state["detect_calls"] == 1  # only the click computes
        assert app.get("download_button")
    finally:  # AppTest shares this interpreter's modules: undo the counting patch
        if hasattr(pages, "_real_detect_features"):
            pages.detect_features = pages._real_detect_features
            del pages._real_detect_features


def test_run_signature_ignores_manual_validation_inputs() -> None:
    import inspect

    from module5_6.webapp.pages import _tracking_signature

    params = set(inspect.signature(_tracking_signature).parameters)
    assert params == {
        "upload", "start_seconds", "frames_to_load", "shi_tomasi_params", "lk_params",
        "validate_fb", "max_fb_error",
    }
