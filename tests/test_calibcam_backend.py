import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nanodeepcharuco.calibcam.backend import (
    build_calibcam_command,
    run_calibcam,
)


class TestCalibCamBackend(unittest.TestCase):
    def test_build_command_uses_calibcam_style_data_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "calibration"

            command = build_calibcam_command(
                python_executable="/opt/calibcam/bin/python",
                videos=[
                    "/data/left.mp4",
                    "/data/right.mp4",
                ],
                detection_paths=[
                    root / "detection_000.yml",
                    root / "detection_001.yml",
                ],
                board="/data/board.npy",
                models=[
                    "omnidir",
                    "omnidir",
                ],
                projection="perspective",
                data_path=root,
            )

            self.assertEqual(
                command,
                [
                    "/opt/calibcam/bin/python",
                    "-m",
                    "calibcam",
                    "--videos",
                    "/data/left.mp4",
                    "/data/right.mp4",
                    "--detection",
                    str(root / "detection_000.yml"),
                    str(root / "detection_001.yml"),
                    "--board",
                    "/data/board.npy",
                    "--calibration_single",
                    "--calibration_multi",
                    "--models",
                    "omnidir",
                    "omnidir",
                    "--projection",
                    "perspective",
                    "--data_path",
                    str(root),
                ],
            )

    def test_mismatched_detection_count_is_rejected(self):
        with self.assertRaises(ValueError):
            build_calibcam_command(
                python_executable="python",
                videos=[
                    "left.mp4",
                    "right.mp4",
                ],
                detection_paths=[
                    "detection_000.yml",
                ],
                board="board.npy",
                models=[
                    "omnidir",
                    "omnidir",
                ],
                projection="perspective",
                data_path="output",
            )

    def test_mismatched_model_count_is_rejected(self):
        with self.assertRaises(ValueError):
            build_calibcam_command(
                python_executable="python",
                videos=[
                    "left.mp4",
                    "right.mp4",
                ],
                detection_paths=[
                    "detection_000.yml",
                    "detection_001.yml",
                ],
                board="board.npy",
                models=[
                    "omnidir",
                ],
                projection="perspective",
                data_path="output",
            )

    @patch(
        "nanodeepcharuco.calibcam.backend.subprocess.run"
    )
    def test_run_calibcam_checks_backend_result(
        self,
        mock_run,
    ):
        command = [
            "python",
            "-m",
            "calibcam",
        ]

        run_calibcam(command)

        mock_run.assert_called_once_with(
            command,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
