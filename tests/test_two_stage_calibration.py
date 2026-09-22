from argparse import Namespace
from pathlib import Path

from nanodeepcharuco.__main__ import run_calibcam


def test_stage1_intrinsics_build_extrinsics_only_command(
    tmp_path,
    monkeypatch,
):
    left_intrinsics = tmp_path / "left_stage1.yml"
    right_intrinsics = tmp_path / "right_stage1.yml"

    left_intrinsics.write_text("left\n")
    right_intrinsics.write_text("right\n")

    detection_left = tmp_path / "detection_000.npy"
    detection_right = tmp_path / "detection_001.npy"

    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(
        "nanodeepcharuco.__main__.subprocess.run",
        fake_run,
    )

    args = Namespace(
        calibcam_python=Path("/tmp/calibcam-python"),
        videos=[
            Path("/tmp/left.MP4"),
            Path("/tmp/right.MP4"),
        ],
        board=Path("/tmp/small_board.npy"),
        models=("omnidir", "omnidir"),
        projection="fisheye_equidistant",
        calibration_single=False,
        calibration_multi=False,
        stage1_intrinsics=(
            left_intrinsics,
            right_intrinsics,
        ),
    )

    run_calibcam(
        args,
        tmp_path / "run",
        (detection_left, detection_right),
    )

    command = captured["command"]

    calibration_single_index = command.index(
        "--calibration_single"
    )

    assert command[
        calibration_single_index + 1:
        calibration_single_index + 3
    ] == [
        str(left_intrinsics),
        str(right_intrinsics),
    ]

    assert "--calibration_multi" in command

    multi_vars_index = command.index("--multi_vars")

    assert command[
        multi_vars_index + 1:
        multi_vars_index + 3
    ] == [
        "extrinsics",
        "extrinsics",
    ]

    assert captured["check"] is True
