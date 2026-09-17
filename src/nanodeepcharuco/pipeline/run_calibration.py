from __future__ import annotations

from pathlib import Path
from typing import Sequence

from nanodeepcharuco.calibcam.backend import (
    build_calibcam_command,
    run_calibcam,
)
from nanodeepcharuco.run_layout import RunLayout


def run_calibration_pipeline(
    *,
    config,
    layout: RunLayout,
    detection_paths: Sequence[str | Path],
) -> None:
    """Run the CalibCam calibration stage."""

    if config.detect_only:
        print(
            "detect_only=True; "
            "CalibCam calibration skipped."
        )
        return

    command = build_calibcam_command(
        python_executable=(
            config.calibcam_python
        ),
        videos=config.videos,
        detection_paths=detection_paths,
        board=config.board,
        models=config.models,
        projection=config.projection,
        data_path=layout.calibcam_data_path,
    )

    print()
    print("=" * 80)
    print("STARTING CALIBCAM BACKEND")
    print("=" * 80)

    print(
        "CalibCam Python:",
        config.calibcam_python,
    )

    print(
        "Detection inputs:",
        [
            str(path)
            for path in detection_paths
        ],
    )

    print(
        "Models:",
        config.models,
    )

    print(
        "Projection:",
        config.projection,
    )

    print(
        "Output:",
        layout.calibcam_data_path,
    )

    print()
    print("Command:")
    print(
        " ".join(
            str(part)
            for part in command
        )
    )
    print()

    run_calibcam(
        command
    )

    print()
    print("=" * 80)
    print("CALIBCAM BACKEND FINISHED")
    print("=" * 80)
    print(
        "Calibration output:",
        layout.calibcam_data_path,
    )
