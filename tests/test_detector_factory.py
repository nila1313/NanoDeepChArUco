import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nanodeepcharuco.pipeline.detectors import (
    build_hybrid_detector,
)


class TestDetectorFactory(unittest.TestCase):
    @patch(
        "nanodeepcharuco.pipeline.detectors."
        "NanoDeepCharucoDetector"
    )
    def test_build_hybrid_detector_forwards_configuration(
        self,
        detector_class,
    ):
        config = SimpleNamespace(
            board="/data/board.npy",
            nano_executable="/opt/nano/detect_batch",
            gamma=1.4,
            deep_self_ransac_px=5.0,
        )

        deep_detector = object()
        work_dir = Path("/tmp/camera_000")

        expected_detector = object()
        detector_class.return_value = (
            expected_detector
        )

        detector = build_hybrid_detector(
            config=config,
            deep_detector=deep_detector,
            work_dir=work_dir,
        )

        self.assertIs(
            detector,
            expected_detector,
        )

        detector_class.assert_called_once_with(
            board_path="/data/board.npy",
            nano_executable="/opt/nano/detect_batch",
            deep_detector=deep_detector,
            work_dir=work_dir,
            gamma=1.4,
            deep_self_ransac_px=5.0,
        )


if __name__ == "__main__":
    unittest.main()
