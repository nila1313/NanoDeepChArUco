import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from nanodeepcharuco.detection.deep_charuco import DeepCharucoDetector


def _stub_detector(mapping, sx, sy, keypoints=None):
    detector = DeepCharucoDetector.__new__(DeepCharucoDetector)
    detector.coordinate_mapping = mapping
    if keypoints is None:
        keypoints = [(10.0, 20.0, 7)]
    detector._infer_keypoints = lambda _frame: (keypoints, sx, sy)
    return detector


@pytest.mark.parametrize(
    ("sx", "sy", "mapping", "expected"),
    [
        (6.0, 4.5, "legacy", (60.0, 90.0)),
        (6.0, 4.5, "centre_aligned", (62.5, 91.75)),
        (6.0, 4.5, "calibrated", (62.36, 91.52)),
        (3.0, 2.25, "legacy", (30.0, 45.0)),
        (3.0, 2.25, "centre_aligned", (31.0, 45.625)),
        (3.0, 2.25, "calibrated", (30.93, 45.51)),
    ],
)
def test_coordinate_mapping_at_two_resolutions(sx, sy, mapping, expected):
    detector = _stub_detector(mapping, sx, sy)
    clean, duplicates, count = detector.detect_with_audit(None)

    assert count == 1
    assert duplicates == []
    assert sorted(clean) == [7]
    assert clean[7].dtype == np.float32
    np.testing.assert_allclose(clean[7], expected, rtol=0, atol=1e-5)


def test_legacy_preserves_original_float32_conversion():
    detector = _stub_detector(
        "legacy", 5.7, 4.3, keypoints=[(12.375, 8.625, 17)]
    )
    clean, _, _ = detector.detect_with_audit(None)
    expected = np.asarray(
        [float(12.375) * 5.7, float(8.625) * 4.3], dtype=np.float32
    )
    np.testing.assert_array_equal(clean[17], expected)


def test_duplicate_ids_remain_rejected():
    detector = _stub_detector(
        "centre_aligned",
        6.0,
        4.5,
        keypoints=[(10.0, 20.0, 7), (11.0, 21.0, 7), (30.0, 40.0, 2)],
    )
    clean, duplicates, count = detector.detect_with_audit(None)
    assert sorted(clean) == [2]
    assert duplicates == [7]
    assert count == 3


def test_default_mapping_is_legacy():
    parameter = inspect.signature(
        DeepCharucoDetector.__init__
    ).parameters["coordinate_mapping"]
    assert parameter.default == "legacy"


def test_invalid_mapping_rejected_before_loading_models():
    with pytest.raises(ValueError, match="Unknown coordinate mapping"):
        DeepCharucoDetector(
            project_root=".",
            deep_checkpoint="does-not-exist.ckpt",
            refinenet_checkpoint="does-not-exist.ckpt",
            coordinate_mapping="incorrect",
        )


def test_calibrated_mapping_rejects_other_network_input_sizes():
    detector = DeepCharucoDetector.__new__(DeepCharucoDetector)
    detector.coordinate_mapping = "calibrated"
    detector.config_path = "unused-config"
    detector._load_configuration = lambda _path: SimpleNamespace(
        input_size=(640, 480)
    )

    with pytest.raises(ValueError, match="320x240"):
        detector.load()
