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
        stage1_calibration=False,
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


def test_stage1_calibration_runs_independent_camera_refinements(
    tmp_path,
    monkeypatch,
):
    detection_left = tmp_path / "detection_000.npy"
    detection_right = tmp_path / "detection_001.npy"

    calls = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(
        "nanodeepcharuco.__main__.subprocess.run",
        fake_run,
    )

    left_video = Path("/tmp/left.MP4")
    right_video = Path("/tmp/right.MP4")

    args = Namespace(
        calibcam_python=Path("/tmp/calibcam-python"),
        videos=[
            left_video,
            right_video,
        ],
        board=Path("/tmp/large_board.npy"),
        models=("omnidir", "omnidir"),
        projection="fisheye_equidistant",
        calibration_single=False,
        calibration_multi=False,
        stage1_calibration=True,
        stage1_intrinsics=None,
    )

    output_root = tmp_path / "run"

    run_calibcam(
        args,
        output_root,
        (detection_left, detection_right),
    )

    assert len(calls) == 2

    expected = (
        (
            "left",
            left_video,
            detection_left,
            "omnidir",
        ),
        (
            "right",
            right_video,
            detection_right,
            "omnidir",
        ),
    )

    for (command, check), (
        side,
        video,
        detection,
        model,
    ) in zip(calls, expected):
        assert check is True

        videos_index = command.index("--videos")
        assert command[videos_index + 1] == str(video)

        detection_index = command.index("--detection")
        assert command[detection_index + 1] == str(detection)

        models_index = command.index("--models")
        assert command[models_index + 1] == model

        data_path_index = command.index("--data_path")
        assert command[data_path_index + 1] == str(
            output_root / "calibcam_output" / side
        )

        assert "--calibration_single" in command
        assert "--calibration_multi" in command
        assert "--multi_vars" not in command
