from argparse import Namespace
from pathlib import Path

import pytest

from nanodeepcharuco.__main__ import run_calibcam
from nanodeepcharuco.calibcam.two_stage_backend import (
    build_parser as build_backend_parser,
    run_stage1 as run_backend_stage1,
    run_stage2 as run_backend_stage2,
)


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

def test_backend_stage1_uses_calibcam_detection_backend(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_run(command, check):
        calls.append((command, check))

        data_path_index = command.index("--data_path")
        output = Path(command[data_path_index + 1])

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output
            / "multicam_calibration.yml"
        ).write_text("calibration\n")

    monkeypatch.setattr(
        "nanodeepcharuco.calibcam.two_stage_backend."
        "subprocess.run",
        fake_run,
    )

    args = Namespace(
        calibcam_python=Path("/tmp/calibcam-python"),
        videos=[
            Path("/tmp/left.MP4"),
            Path("/tmp/right.MP4"),
        ],
        board=Path("/tmp/large_board.npy"),
        project_root=tmp_path / "project",
        nano_executable=tmp_path / "detect_batch",
        deep_checkpoint=tmp_path / "detector.ckpt",
        refinenet_checkpoint=tmp_path / "refinenet.ckpt",
        config_path=tmp_path / "large.yaml",
        device="cpu",
        frames_start=0,
        frames_end=100,
        frames_step=20,
        model="omnidir",
        projection="fisheye_equidistant",
        data_path=tmp_path / "stage1",
        keep_work=False,
    )

    run_backend_stage1(args)

    assert len(calls) == 2

    for command, check in calls:
        assert check is True

        assert "--opts" in command
        assert "--detection" in command
        assert "--calibration_single" in command
        assert "--calibration_multi" in command
        assert "--multi_vars" not in command

        detection_index = command.index(
            "--detection"
        )

        assert command[
            detection_index + 1
        ] == "--calibration_single"

        assert not any(
            item.endswith(".npy")
            and "detection_" in item
            for item in command
        )

        opts_index = command.index("--opts")
        opts_path = Path(
            command[opts_index + 1]
        )

        assert opts_path.is_file()
        assert (
            "backend: nanodeepcharuco"
            in opts_path.read_text()
        )


def test_backend_stage2_uses_fixed_intrinsics_and_offsets(
    tmp_path,
    monkeypatch,
):
    left_intrinsics = (
        tmp_path / "left_stage1.yml"
    )
    right_intrinsics = (
        tmp_path / "right_stage1.yml"
    )

    left_intrinsics.write_text("left\n")
    right_intrinsics.write_text("right\n")

    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

        data_path_index = command.index("--data_path")
        output = Path(command[data_path_index + 1])

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output
            / "multicam_calibration.yml"
        ).write_text("calibration\n")

    monkeypatch.setattr(
        "nanodeepcharuco.calibcam.two_stage_backend."
        "subprocess.run",
        fake_run,
    )

    args = Namespace(
        calibcam_python=Path("/tmp/calibcam-python"),
        videos=[
            Path("/tmp/left.MP4"),
            Path("/tmp/right.MP4"),
        ],
        board=Path("/tmp/small_board.npy"),
        stage1_intrinsics=(
            left_intrinsics,
            right_intrinsics,
        ),
        project_root=tmp_path / "project",
        nano_executable=tmp_path / "detect_batch",
        deep_checkpoint=tmp_path / "detector.ckpt",
        refinenet_checkpoint=tmp_path / "refinenet.ckpt",
        config_path=tmp_path / "small.yaml",
        device="cpu",
        frames_start=0,
        frames_end=2640,
        frames_step=20,
        frames_offsets=(0, 1),
        models=("omnidir", "omnidir"),
        projection="fisheye_equidistant",
        data_path=tmp_path / "stage2",
        keep_work=False,
    )

    run_backend_stage2(args)

    command = captured["command"]

    assert captured["check"] is True
    assert "--opts" in command
    assert "--detection" in command
    assert "--calibration_multi" in command

    detection_index = command.index(
        "--detection"
    )

    assert command[
        detection_index + 1
    ] == "--calibration_single"

    calibration_index = command.index(
        "--calibration_single"
    )

    assert command[
        calibration_index + 1:
        calibration_index + 3
    ] == [
        str(left_intrinsics),
        str(right_intrinsics),
    ]

    multi_vars_index = command.index(
        "--multi_vars"
    )

    assert command[
        multi_vars_index + 1:
        multi_vars_index + 3
    ] == [
        "extrinsics",
        "extrinsics",
    ]

    offsets_index = command.index(
        "--frames_offsets"
    )

    assert command[
        offsets_index + 1:
        offsets_index + 3
    ] == [
        "0",
        "1",
    ]

    assert not any(
        item.endswith(".npy")
        and "detection_" in item
        for item in command
    )


def test_backend_stage2_requires_explicit_frame_offsets(
    tmp_path,
):
    def touch(name):
        path = tmp_path / name
        path.write_text("x\n")
        return path

    left_video = touch("left.MP4")
    right_video = touch("right.MP4")
    board = touch("small_board.npy")
    left_intrinsics = touch("left_stage1.yml")
    right_intrinsics = touch("right_stage1.yml")
    nano_executable = touch("detect_batch")
    deep_checkpoint = touch("detector.ckpt")
    refinenet_checkpoint = touch("refinenet.ckpt")
    config_path = touch("small.yaml")

    parser = build_backend_parser()

    argv = [
        "stage2",
        "--videos",
        str(left_video),
        str(right_video),
        "--board",
        str(board),
        "--stage1-intrinsics",
        str(left_intrinsics),
        str(right_intrinsics),
        "--nano-executable",
        str(nano_executable),
        "--deep-checkpoint",
        str(deep_checkpoint),
        "--refinenet-checkpoint",
        str(refinenet_checkpoint),
        "--config-path",
        str(config_path),
        "--data-path",
        str(tmp_path / "output"),
    ]

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)

    assert exc.value.code == 2
