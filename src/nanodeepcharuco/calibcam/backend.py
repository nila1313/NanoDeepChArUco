from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def build_calibcam_command(
    *,
    python_executable: str,
    videos: Sequence[str],
    detection_paths: Sequence[str | Path],
    board: str,
    models: Sequence[str],
    projection: str,
    data_path: str | Path,
) -> list[str]:
    """Build the command used to run the CalibCam backend."""

    if len(videos) != len(detection_paths):
        raise ValueError(
            "CalibCam requires one detection file per video."
        )

    if len(videos) != len(models):
        raise ValueError(
            "CalibCam requires one camera model per video."
        )

    return [
        str(python_executable),
        "-m",
        "calibcam",
        "--videos",
        *[str(video) for video in videos],
        "--detection",
        *[
            str(Path(path))
            for path in detection_paths
        ],
        "--board",
        str(board),
        "--calibration_single",
        "--calibration_multi",
        "--models",
        *[str(model) for model in models],
        "--projection",
        str(projection),
        "--data_path",
        str(Path(data_path)),
    ]


def run_calibcam(
    command: Sequence[str],
) -> None:
    """Run CalibCam and fail if the backend exits unsuccessfully."""

    subprocess.run(
        list(command),
        check=True,
    )
