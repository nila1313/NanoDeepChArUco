import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nanodeepcharuco.pipeline.run_calibration import (
    run_calibration_pipeline,
)


class FakeLayout:
    calibcam_data_path = Path(
        "/tmp/run"
    )


class TestCalibrationWorkflow(unittest.TestCase):
    @patch(
        "nanodeepcharuco.pipeline.run_calibration."
        "run_calibcam"
    )
    @patch(
        "nanodeepcharuco.pipeline.run_calibration."
        "build_calibcam_command"
    )
    def test_calibration_builds_and_runs_backend(
        self,
        build_command,
        run_backend,
    ):
        config = SimpleNamespace(
            detect_only=False,
            calibcam_python="/opt/calibcam/python",
            videos=[
                "left.mp4",
                "right.mp4",
            ],
            board="board.npy",
            models=[
                "omnidir",
                "omnidir",
            ],
            projection="perspective",
        )

        layout = FakeLayout()

        detection_paths = [
            Path(
                "/tmp/run/detection_000.yml"
            ),
            Path(
                "/tmp/run/detection_001.yml"
            ),
        ]

        command = [
            "/opt/calibcam/python",
            "-m",
            "calibcam",
        ]

        build_command.return_value = command

        with redirect_stdout(StringIO()):
            run_calibration_pipeline(
                config=config,
                layout=layout,
                detection_paths=detection_paths,
            )

        build_command.assert_called_once_with(
            python_executable=(
                "/opt/calibcam/python"
            ),
            videos=[
                "left.mp4",
                "right.mp4",
            ],
            detection_paths=detection_paths,
            board="board.npy",
            models=[
                "omnidir",
                "omnidir",
            ],
            projection="perspective",
            data_path=(
                layout.calibcam_data_path
            ),
        )

        run_backend.assert_called_once_with(
            command
        )

    @patch(
        "nanodeepcharuco.pipeline.run_calibration."
        "run_calibcam"
    )
    @patch(
        "nanodeepcharuco.pipeline.run_calibration."
        "build_calibcam_command"
    )
    def test_detect_only_skips_calibration(
        self,
        build_command,
        run_backend,
    ):
        config = SimpleNamespace(
            detect_only=True,
        )

        with redirect_stdout(StringIO()):
            run_calibration_pipeline(
                config=config,
                layout=FakeLayout(),
                detection_paths=[],
            )

        build_command.assert_not_called()
        run_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
