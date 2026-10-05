from pathlib import Path

import yaml

from nanodeepcharuco.__main__ import parse_args


def touch(path: Path) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text("test\n")
    return path


def test_pipeline_config_supplies_defaults(
    tmp_path,
):
    left = touch(
        tmp_path / "left.MP4"
    )

    right = touch(
        tmp_path / "right.MP4"
    )

    board = touch(
        tmp_path / "board.npy"
    )

    config = tmp_path / "pipeline.yaml"

    config.write_text(
        yaml.safe_dump(
            {
                "board": str(board),
                "detector": "hybrid",
                "auto_sync": True,
                "stable_motion_mode": (
                    "adaptive"
                ),
                "frames_step": 20,
            }
        )
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--videos",
            str(left),
            str(right),
            "--data_path",
            str(tmp_path / "run"),
        ]
    )

    assert args.board == board.resolve()
    assert args.detector == "hybrid"
    assert args.auto_sync is True
    assert (
        args.stable_motion_mode
        == "adaptive"
    )


def test_cli_overrides_pipeline_config(
    tmp_path,
):
    left = touch(
        tmp_path / "left.MP4"
    )

    right = touch(
        tmp_path / "right.MP4"
    )

    board = touch(
        tmp_path / "board.npy"
    )

    config = tmp_path / "pipeline.yaml"

    config.write_text(
        yaml.safe_dump(
            {
                "board": str(board),
                "frames_step": 20,
                "stable_motion_mode": (
                    "adaptive"
                ),
            }
        )
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--videos",
            str(left),
            str(right),
            "--frames_step",
            "10",
            "--stable_motion_mode",
            "fixed",
            "--data_path",
            str(tmp_path / "run"),
        ]
    )

    assert args.frames_step == 10
    assert (
        args.stable_motion_mode
        == "fixed"
    )



def test_pipeline_config_supplies_calibcam_board(
    tmp_path,
):
    left = touch(tmp_path / "left.MP4")
    right = touch(tmp_path / "right.MP4")
    board = touch(tmp_path / "board.npy")
    calibcam_board = touch(
        tmp_path / "board_numpy1.npy"
    )

    config = tmp_path / "pipeline.yaml"

    config.write_text(
        yaml.safe_dump(
            {
                "board": str(board),
                "calibcam_board": str(
                    calibcam_board
                ),
            }
        )
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--videos",
            str(left),
            str(right),
            "--data_path",
            str(tmp_path / "run"),
        ]
    )

    assert args.board == board.resolve()
    assert (
        args.calibcam_board
        == calibcam_board.resolve()
    )
