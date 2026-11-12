from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from nanodeepcharuco.calibcam_backend import (
    NanoDeepCharucoBackend,
    _nanodeep_frame_prefix,
    _normalize_nanodeep_config_paths,
)


def test_frame_conversion_sorts_ids_and_preserves_tag():
    detector = Mock()
    detector.process_frame.return_value = SimpleNamespace(
        corners={5: [50.0, 60.0], 2: [20.0, 30.0]}
    )
    context = {"detector": detector, "frame_prefix": "pair01"}
    frame = np.zeros((4, 5, 3), dtype=np.uint8)

    corners, ids = NanoDeepCharucoBackend().detect_frame(frame, 21, context)

    assert corners.shape == (2, 1, 2)
    assert corners.dtype == np.float32
    assert ids.shape == (2, 1)
    assert ids.dtype == np.int32
    assert ids.tolist() == [[2], [5]]
    np.testing.assert_allclose(
        corners[:, 0, :],
        [[20.0, 30.0], [50.0, 60.0]],
    )
    detector.process_frame.assert_called_once_with(frame, "pair01_00000021")


def test_empty_detection_returns_none():
    detector = Mock()
    detector.process_frame.return_value = SimpleNamespace(corners={})
    context = {"detector": detector, "frame_prefix": "pair01"}
    frame = np.zeros((4, 5, 3), dtype=np.uint8)

    corners, ids = NanoDeepCharucoBackend().detect_frame(frame, 0, context)
    assert corners is None
    assert ids is None


def test_frame_prefix_distinguishes_same_filenames():
    first = _nanodeep_frame_prefix("/tmp/dataset_a/camera.mp4")
    second = _nanodeep_frame_prefix("/tmp/dataset_b/camera.mp4")
    assert first.startswith("camera_")
    assert second.startswith("camera_")
    assert first != second


def test_config_paths_normalized_without_mutating_input():
    config = {"project_root": Path("/tmp/nanodeep"), "keep_work": False}
    original = dict(config)
    normalized = _normalize_nanodeep_config_paths(config)

    assert config == original
    assert isinstance(config["project_root"], Path)
    assert normalized is not config
    assert normalized["project_root"] == "/tmp/nanodeep"
    assert normalized["keep_work"] is False


def test_backend_default_options_are_complete():
    expected = {
        "project_root": False,
        "nano_executable": False,
        "deep_checkpoint": False,
        "refinenet_checkpoint": False,
        "config_path": False,
        "device": False,
        "work_dir": False,
        "keep_work": False,
    }
    assert NanoDeepCharucoBackend.default_opts == expected


def test_missing_required_options_raise_without_loading_models():
    import pytest
    from nanodeepcharuco.calibcam_backend import _build_nanodeep_detector

    with pytest.raises(ValueError, match="missing required options"):
        _build_nanodeep_detector(board=object(), config={})


def test_backend_receives_raw_frames():
    """NanoDeepCharuco was validated on raw frames, so calibcam must not preprocess them."""
    assert NanoDeepCharucoBackend.uses_preprocessing is False


def test_create_context_fills_defaults_locally_without_changing_opts():
    from copy import deepcopy
    from unittest.mock import patch

    opts = {"detection_opts": {"nanodeepcharuco": {"work_dir": Path("/tmp/work")}}}
    original = deepcopy(opts)

    with patch(
        "nanodeepcharuco.calibcam_backend._build_nanodeep_detector",
        return_value="detector",
    ) as build:
        context = NanoDeepCharucoBackend().create_context("/tmp/left.mp4", object(), opts)

    config = build.call_args.args[1]
    assert config["work_dir"] == Path("/tmp/work")
    assert config["keep_work"] is False  # filled from default_opts
    assert set(config) == set(NanoDeepCharucoBackend.default_opts)
    assert context["detector"] == "detector"
    assert opts == original
