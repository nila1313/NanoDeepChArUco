from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()

    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"File not found: {path}"
        )

    return path


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def write_nanodeep_opts(
    path: Path,
    *,
    project_root: Path,
    nano_executable: Path,
    deep_checkpoint: Path,
    refinenet_checkpoint: Path,
    config_path: Path,
    device: str | None,
    work_dir: Path,
    keep_work: bool,
) -> Path:
    opts = {
        "detection_opts": {
            "backend": "nanodeepcharuco",
            "nanodeepcharuco": {
                "project_root": str(
                    resolved(project_root)
                ),
                "nano_executable": str(
                    resolved(nano_executable)
                ),
                "deep_checkpoint": str(
                    resolved(deep_checkpoint)
                ),
                "refinenet_checkpoint": str(
                    resolved(refinenet_checkpoint)
                ),
                "config_path": str(
                    resolved(config_path)
                ),
                "device": (
                    device
                    if device is not None
                    else False
                ),
                "work_dir": str(
                    resolved(work_dir)
                ),
                "keep_work": bool(keep_work),
            },
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            opts,
            f,
            sort_keys=False,
        )

    return path


def run_stage1(args: argparse.Namespace) -> tuple[Path, Path]:
    output_root = resolved(args.data_path)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibcam_python = str(
        resolved(args.calibcam_python)
    )

    results: list[Path] = []

    for side, video in zip(
        ("left", "right"),
        args.videos,
    ):
        side_output = (
            output_root
            / side
        )

        side_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        work_dir = (
            output_root
            / "nanodeep_work"
            / side
        )

        opts_path = (
            output_root
            / f"nanodeep_opts_{side}.yml"
        )

        write_nanodeep_opts(
            opts_path,
            project_root=args.project_root,
            nano_executable=args.nano_executable,
            deep_checkpoint=args.deep_checkpoint,
            refinenet_checkpoint=args.refinenet_checkpoint,
            config_path=args.config_path,
            device=args.device,
            work_dir=work_dir,
            keep_work=args.keep_work,
        )

        command = [
            calibcam_python,
            "-m",
            "calibcam",
            "--videos",
            str(resolved(video)),
            "--board",
            str(resolved(args.board)),
            "--opts",
            str(opts_path),
            "--detection",
            "--calibration_single",
            "--calibration_multi",
            "--models",
            args.model,
            "--projection",
            args.projection,
            "--frames_start",
            str(args.frames_start),
            "--frames_step",
            str(args.frames_step),
            "--data_path",
            str(side_output),
        ]

        if args.frames_end is not None:
            command.extend(
                [
                    "--frames_end",
                    str(args.frames_end),
                ]
            )

        print()
        print(
            f"Running Stage 1 for {side} camera"
        )
        print(
            " ".join(command)
        )
        print()

        subprocess.run(
            command,
            check=True,
        )

        calibration_path = (
            side_output
            / "multicam_calibration.yml"
        )

        if not calibration_path.is_file():
            raise FileNotFoundError(
                "Expected Stage-1 calibration was not "
                f"created: {calibration_path}"
            )

        results.append(
            calibration_path
        )

    left_calibration, right_calibration = results

    print()
    print("Stage 1 completed")
    print("-----------------")
    print(
        "Left intrinsics :",
        left_calibration,
    )
    print(
        "Right intrinsics:",
        right_calibration,
    )

    return (
        left_calibration,
        right_calibration,
    )



def run_stage2(args: argparse.Namespace) -> Path:
    output_root = resolved(args.data_path)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibcam_python = str(
        resolved(args.calibcam_python)
    )

    work_dir = (
        output_root
        / "nanodeep_work"
    )

    opts_path = (
        output_root
        / "nanodeep_opts_stage2.yml"
    )

    write_nanodeep_opts(
        opts_path,
        project_root=args.project_root,
        nano_executable=args.nano_executable,
        deep_checkpoint=args.deep_checkpoint,
        refinenet_checkpoint=args.refinenet_checkpoint,
        config_path=args.config_path,
        device=args.device,
        work_dir=work_dir,
        keep_work=args.keep_work,
    )

    command = [
        calibcam_python,
        "-m",
        "calibcam",
        "--videos",
        str(resolved(args.videos[0])),
        str(resolved(args.videos[1])),
        "--board",
        str(resolved(args.board)),
        "--opts",
        str(opts_path),
        "--detection",
        "--calibration_single",
        str(resolved(args.stage1_intrinsics[0])),
        str(resolved(args.stage1_intrinsics[1])),
        "--calibration_multi",
        "--multi_vars",
        "extrinsics",
        "extrinsics",
        "--models",
        args.models[0],
        args.models[1],
        "--projection",
        args.projection,
        "--frames_start",
        str(args.frames_start),
        "--frames_step",
        str(args.frames_step),
        "--frames_offsets",
        str(args.frames_offsets[0]),
        str(args.frames_offsets[1]),
        "--data_path",
        str(output_root),
    ]

    if args.frames_end is not None:
        command.extend(
            [
                "--frames_end",
                str(args.frames_end),
            ]
        )

    print()
    print("Running Stage 2 stereo calibration")
    print("----------------------------------")
    print(
        "Left offset :",
        args.frames_offsets[0],
    )
    print(
        "Right offset:",
        args.frames_offsets[1],
    )
    print()
    print(
        " ".join(command)
    )
    print()

    subprocess.run(
        command,
        check=True,
    )

    calibration_path = (
        output_root
        / "multicam_calibration.yml"
    )

    if not calibration_path.is_file():
        raise FileNotFoundError(
            "Expected Stage-2 calibration was not "
            f"created: {calibration_path}"
        )

    print()
    print("Stage 2 completed")
    print("-----------------")
    print(
        "Stereo calibration:",
        calibration_path,
    )

    return calibration_path


def add_common_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--videos",
        nargs=2,
        required=True,
        type=existing_file,
        metavar=("LEFT", "RIGHT"),
    )

    parser.add_argument(
        "--board",
        required=True,
        type=existing_file,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )

    parser.add_argument(
        "--nano-executable",
        required=True,
        type=existing_file,
    )

    parser.add_argument(
        "--deep-checkpoint",
        required=True,
        type=existing_file,
    )

    parser.add_argument(
        "--refinenet-checkpoint",
        required=True,
        type=existing_file,
    )

    parser.add_argument(
        "--config-path",
        required=True,
        type=existing_file,
    )

    parser.add_argument(
        "--device",
        choices=(
            "cpu",
            "cuda",
            "mps",
        ),
        default=None,
    )

    parser.add_argument(
        "--frames-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--frames-end",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--frames-step",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--projection",
        default="fisheye_equidistant",
    )

    parser.add_argument(
        "--calibcam-python",
        type=existing_file,
        default=Path(sys.executable),
    )

    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--keep-work",
        action="store_true",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "nanodeepcharuco.calibcam.two_stage_backend"
        ),
        description=(
            "Wide-angle two-stage calibration using "
            "NanoDeepCharuco through the CalibCam "
            "detection backend."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    stage1 = subparsers.add_parser(
        "stage1",
        help=(
            "Estimate left and right intrinsics "
            "independently from large-board videos."
        ),
    )

    add_common_arguments(stage1)

    stage1.add_argument(
        "--model",
        default="omnidir",
    )

    stage2 = subparsers.add_parser(
        "stage2",
        help=(
            "Estimate stereo extrinsics from small-board "
            "videos while keeping Stage-1 intrinsics fixed."
        ),
    )

    add_common_arguments(stage2)

    stage2.add_argument(
        "--stage1-intrinsics",
        nargs=2,
        required=True,
        type=existing_file,
        metavar=(
            "LEFT_CALIBRATION",
            "RIGHT_CALIBRATION",
        ),
    )

    stage2.add_argument(
        "--frames-offsets",
        nargs=2,
        type=int,
        required=True,
        metavar=("LEFT", "RIGHT"),
        help=(
            "Explicit left and right frame offsets. "
            "Synchronization is determined outside "
            "this calibration runner."
        ),
    )

    stage2.add_argument(
        "--models",
        nargs=2,
        default=("omnidir", "omnidir"),
        metavar=("LEFT", "RIGHT"),
    )

    return parser

def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    if args.command == "stage1":
        run_stage1(args)
        return 0

    if args.command == "stage2":
        run_stage2(args)
        return 0

    parser.error(
        f"Unsupported command: {args.command}"
    )

    return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
