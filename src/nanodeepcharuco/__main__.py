from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return number


def load_config_defaults(
    argv: list[str] | None,
) -> tuple[Path | None, dict]:
    """
    Read --config before parsing the main CLI.

    Values from the YAML file become defaults. Explicit command-line
    arguments parsed later always take precedence.
    """

    config_parser = argparse.ArgumentParser(
        add_help=False,
    )

    config_parser.add_argument(
        "--config",
        type=existing_file,
        default=None,
    )

    config_args, _ = config_parser.parse_known_args(
        argv,
    )

    if config_args.config is None:
        return None, {}

    config_path = config_args.config

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError(
            "Pipeline config must contain a YAML mapping "
            "at the top level"
        )

    return config_path, config



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_path, config_defaults = load_config_defaults(
        argv,
    )

    parser = argparse.ArgumentParser(
        prog="nanodeepcharuco",
        description=(
            "Generate CalibCam-compatible stereo ChArUco detections with "
            "OpenCV, ArUco Nano, or the NanoDeepCharuco hybrid detector."
        ),
    )
    parser.add_argument(
        "--config",
        type=existing_file,
        default=config_path,
        help=(
            "YAML pipeline profile. Values from the profile become "
            "defaults; explicit command-line options override them."
        ),
    )

    parser.add_argument(
        "--videos",
        nargs=2,
        required=("videos" not in config_defaults),
        type=existing_file,
        metavar=("LEFT", "RIGHT"),
    )
    parser.add_argument(
        "--board",
        required=("board" not in config_defaults),
        type=existing_file,
    )
    parser.add_argument("--detector", choices=("opencv", "nano", "hybrid"),
                        default="nano")
    parser.add_argument("--frames_start", type=int, default=0)
    parser.add_argument("--frames_end", type=int, default=None)
    parser.add_argument("--frames_step", type=positive_int, default=20)
    parser.add_argument("--frames_offsets", "--frames_offset", nargs=2, type=int,
                        default=(0, 0), dest="frames_offsets",
                        metavar=("LEFT_OFFSET", "RIGHT_OFFSET"))

    parser.add_argument(
        "--auto_sync",
        action="store_true",
        help=(
            "Automatically estimate trusted stereo synchronization "
            "for two-camera Stage-2 calibration."
        ),
    )
    parser.add_argument(
        "--sync_offsets",
        nargs="+",
        type=int,
        default=(-3, -2, -1, 0, 1, 2, 3),
        help=(
            "Candidate right-camera offsets for automatic synchronization."
        ),
    )
    parser.add_argument(
        "--sync_frame_step",
        type=positive_int,
        default=1,
        help=(
            "Physical-frame sampling step used during synchronization discovery."
        ),
    )
    parser.add_argument(
        "--sync_window_size",
        type=positive_int,
        default=400,
        help="Synchronization window size in frames.",
    )
    parser.add_argument(
        "--sync_window_step",
        type=positive_int,
        default=200,
        help="Synchronization window step in frames.",
    )
    parser.add_argument(
        "--sync_min_inlier_ratio",
        type=float,
        default=0.20,
        help="Minimum epipolar inlier ratio for a trusted sync window.",
    )
    parser.add_argument(
        "--sync_min_ratio_gap",
        type=float,
        default=0.05,
        help=(
            "Minimum inlier-ratio advantage over the second-best offset."
        ),
    )
    parser.add_argument(
        "--sync_min_persistence",
        type=positive_int,
        default=2,
        help="Minimum number of supporting windows for a stable sync segment.",
    )
    parser.add_argument(
        "--sync_max_gap_frames",
        type=int,
        default=400,
        help=(
            "Maximum frame gap allowed between supporting windows "
            "of the same offset."
        ),
    )

    parser.add_argument(
        "--stable_motion_px",
        type=float,
        default=2.0,
        help=(
            "Maximum median ChArUco-corner motion in pixels "
            "for a frame transition to count as board-stable."
        ),
    )
    parser.add_argument(
        "--stable_motion_mode",
        choices=("fixed", "adaptive"),
        default="fixed",
        help=(
            "Board-stability threshold mode. "
            "'fixed' uses --stable_motion_px directly. "
            "'adaptive' derives a threshold from the observed "
            "motion distribution, with --stable_motion_px as "
            "the minimum and --stable_motion_max_px as the cap."
        ),
    )
    parser.add_argument(
        "--stable_motion_percentile",
        type=float,
        default=25.0,
        help=(
            "Motion percentile used in adaptive board-stability "
            "mode."
        ),
    )
    parser.add_argument(
        "--stable_motion_max_px",
        type=float,
        default=6.0,
        help=(
            "Hard upper limit for the automatically selected "
            "board-motion threshold."
        ),
    )
    parser.add_argument(
        "--stable_min_pairs",
        type=positive_int,
        default=8,
        help=(
            "Minimum number of stereo pairs required after "
            "board-stability filtering."
        ),
    )

    parser.add_argument(
        "--stable_min_frames",
        type=positive_int,
        default=3,
        help=(
            "Minimum number of consecutive physical frames "
            "required for a stable board interval."
        ),
    )
    parser.add_argument("--models", "--model", nargs=2,
                        default=("omnidir", "omnidir"), dest="models",
                        metavar=("LEFT_MODEL", "RIGHT_MODEL"))
    parser.add_argument("--projection", default="perspective")
    parser.add_argument(
        "--data_path",
        required=("data_path" not in config_defaults),
        type=Path,
    )
    parser.add_argument("--nano_executable", type=Path,
                        default=PROJECT_ROOT / "third_party/aruco_nano/build/detect_batch")
    parser.add_argument("--deep_checkpoint", type=Path, default=None)
    parser.add_argument("--refinenet_checkpoint", type=Path,
                        default=PROJECT_ROOT / "models/refinenet/refinenet.ckpt")
    parser.add_argument("--deep_config", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--keep_work",
        action="store_true",
        help=(
            "Keep temporary per-frame Nano detector directories for "
            "debugging. By default they are removed after detection."
        ),
    )
    parser.add_argument("--run_calibcam", action="store_true",
                        help="Run CalibCam after generating detection payloads.")
    parser.add_argument("--calibration_single", action="store_true",
                        help="Backward-compatible alias that enables CalibCam.")
    parser.add_argument("--calibration_multi", action="store_true",
                        help="Backward-compatible alias that enables CalibCam.")
    parser.add_argument(
        "--stage1_calibration",
        action="store_true",
        help=(
            "Run Stage 1 of the wide-angle two-stage workflow: "
            "estimate independent camera intrinsics from the large board."
        ),
    )
    parser.add_argument(
        "--stage1_intrinsics",
        nargs=2,
        type=existing_file,
        default=None,
        metavar=("LEFT_YML", "RIGHT_YML"),
        help=(
            "Stage-1 single-camera calibration files. "
            "When provided, CalibCam keeps these intrinsics fixed "
            "and optimizes stereo extrinsics only."
        ),
    )
    parser.add_argument("--calibcam_python", type=Path,
                        default=Path(sys.executable))
    parser.add_argument(
        "--calibcam_board",
        type=Path,
        default=None,
        help=(
            "Optional CalibCam-compatible serialization of the "
            "same physical board. If omitted, --board is used."
        ),
    )
    parser.add_argument(
        "--calibcam_reject_outliers",
        action="store_true",
        help=(
            "After Stage-2 CalibCam calibration, reject clearly "
            "bad stereo pairs using final CalibCam residuals and "
            "rerun the extrinsic calibration once."
        ),
    )
    parser.add_argument(
        "--calibcam_outlier_mad_multiplier",
        type=float,
        default=6.0,
        help=(
            "MAD multiplier for Stage-2 calibration outlier "
            "detection."
        ),
    )
    parser.add_argument(
        "--calibcam_outlier_min_px",
        type=float,
        default=6.0,
        help=(
            "Minimum residual threshold for automatic "
            "Stage-2 pair rejection."
        ),
    )
    parser.add_argument(
        "--calibcam_outlier_max_px",
        type=float,
        default=10.0,
        help=(
            "Maximum residual threshold for automatic "
            "Stage-2 pair rejection."
        ),
    )

    valid_config_keys = {
        action.dest
        for action in parser._actions
        if action.dest
        not in {
            "help",
            "config",
        }
    }

    unknown_config_keys = (
        set(config_defaults)
        - valid_config_keys
    )

    if unknown_config_keys:
        parser.error(
            "Unknown config option(s): "
            + ", ".join(
                sorted(
                    unknown_config_keys
                )
            )
        )

    parser.set_defaults(
        **config_defaults
    )

    args = parser.parse_args(argv)

    if args.calibcam_outlier_mad_multiplier < 0:
        parser.error(
            "--calibcam_outlier_mad_multiplier "
            "cannot be negative"
        )

    if args.calibcam_outlier_min_px < 0:
        parser.error(
            "--calibcam_outlier_min_px cannot be negative"
        )

    if (
        args.calibcam_outlier_max_px
        < args.calibcam_outlier_min_px
    ):
        parser.error(
            "--calibcam_outlier_max_px cannot be smaller "
            "than --calibcam_outlier_min_px"
        )

    if args.stage1_calibration and args.stage1_intrinsics is not None:
        parser.error(
            "--stage1_calibration and --stage1_intrinsics "
            "cannot be used together"
        )

    if args.auto_sync and args.stage1_calibration:
        parser.error(
            "--auto_sync is intended for Stage 2 stereo calibration, "
            "not independent Stage-1 intrinsic calibration"
        )

    if args.auto_sync and args.sync_max_gap_frames < 0:
        parser.error("--sync_max_gap_frames cannot be negative")

    if args.auto_sync and args.stable_motion_px < 0:
        parser.error("--stable_motion_px cannot be negative")

    if args.auto_sync and args.stable_min_frames < 2:
        parser.error("--stable_min_frames must be at least 2")

    if (
        args.auto_sync
        and not 0.0
        <= args.stable_motion_percentile
        <= 100.0
    ):
        parser.error(
            "--stable_motion_percentile must be between "
            "0 and 100"
        )

    if (
        args.auto_sync
        and args.stable_motion_max_px
        < args.stable_motion_px
    ):
        parser.error(
            "--stable_motion_max_px cannot be smaller than "
            "--stable_motion_px"
        )

    if args.auto_sync and not args.sync_offsets:
        parser.error("--sync_offsets must contain at least one candidate offset")

    return args


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def make_detector(args: argparse.Namespace, side: str, output_root: Path):
    if args.detector == "opencv":
        from nanodeepcharuco.detection.opencv_charuco import OpenCVCharucoDetector
        return OpenCVCharucoDetector(args.board)

    nano_executable = resolved(args.nano_executable)
    work_dir = output_root / "work" / side
    if args.detector == "nano":
        from nanodeepcharuco.detection.nano_charuco import NanoCharucoDetector
        return NanoCharucoDetector(
            args.board,
            nano_executable,
            work_dir,
            keep_work=args.keep_work,
        )

    if args.deep_checkpoint is None:
        raise ValueError("--deep_checkpoint is required for the hybrid detector")

    if args.deep_config is None:
        raise ValueError("--deep_config is required for the hybrid detector")
    from nanodeepcharuco.detection.deep_charuco import DeepCharucoDetector
    from nanodeepcharuco.detection.hybrid import NanoDeepCharucoDetector
    deep = DeepCharucoDetector(
        project_root=PROJECT_ROOT,
        deep_checkpoint=resolved(args.deep_checkpoint),
        refinenet_checkpoint=resolved(args.refinenet_checkpoint),
        config_path=resolved(args.deep_config),
        device=args.device,
    ).load()
    return NanoDeepCharucoDetector(
        args.board,
        nano_executable,
        deep,
        work_dir,
        keep_work=args.keep_work,
    )


def run_calibcam(args: argparse.Namespace, output_root: Path,
                 detection_paths: tuple[Path, Path]) -> None:
    calibcam_output = output_root / "calibcam_output"
    calibcam_python = str(resolved(args.calibcam_python))

    calibcam_board_arg = getattr(
        args,
        "calibcam_board",
        None,
    )

    calibcam_board = resolved(
        calibcam_board_arg
        if calibcam_board_arg is not None
        else args.board
    )

    if getattr(args, "stage1_calibration", False):
        for side, video, detection, model in zip(
            ("left", "right"),
            args.videos,
            detection_paths,
            args.models,
        ):
            side_output = calibcam_output / side
            side_output.mkdir(parents=True, exist_ok=True)

            command = [
                calibcam_python,
                "-m",
                "calibcam",
                "--videos",
                str(video),
                "--detection",
                str(detection),
                "--board",
                str(calibcam_board),
                "--models",
                model,
                "--projection",
                args.projection,
                "--data_path",
                str(side_output),
                "--calibration_single",
                "--calibration_multi",
            ]

            subprocess.run(command, check=True)

        return

    command = [
        calibcam_python,
        "-m",
        "calibcam",
        "--videos",
        *(str(path) for path in args.videos),
        "--detection",
        *(str(path) for path in detection_paths),
        "--board",
        str(calibcam_board),
        "--models",
        *args.models,
        "--projection",
        args.projection,
        "--data_path",
        str(calibcam_output),
    ]

    if args.stage1_intrinsics is not None:
        command.extend([
            "--calibration_single",
            *(str(path) for path in args.stage1_intrinsics),
            "--calibration_multi",
            "--multi_vars",
            "extrinsics",
            "extrinsics",
        ])
    else:
        if args.calibration_single:
            command.append("--calibration_single")
        if args.calibration_multi:
            command.append("--calibration_multi")

    subprocess.run(command, check=True)

    if (
        args.stage1_intrinsics is not None
        and getattr(
            args,
            "calibcam_reject_outliers",
            False,
        )
    ):
        import shutil

        from nanodeepcharuco.calibcam.outlier_rejection import (
            analyze_board_positions,
            compare_extrinsics,
            save_report,
            write_cleaned_detection_payloads,
        )

        board_positions = (
            calibcam_output
            / "multicam_calibration_board_positions.yml"
        )

        initial_calibration = (
            calibcam_output
            / "multicam_calibration.yml"
        )

        report = analyze_board_positions(
            board_positions,
            mad_multiplier=(
                args.calibcam_outlier_mad_multiplier
            ),
            min_threshold_px=(
                args.calibcam_outlier_min_px
            ),
            max_threshold_px=(
                args.calibcam_outlier_max_px
            ),
        )

        report_path = (
            output_root
            / "calibcam_outlier_report.json"
        )

        print()
        print("CalibCam Stage-2 outlier analysis")
        print("---------------------------------")
        print(
            "pairs analyzed :",
            report["n_pairs"],
        )
        print(
            "median pair max:",
            f'{report["median_pair_max_px"]:.3f}px',
        )
        print(
            "MAD            :",
            f'{report["mad_pair_max_px"]:.3f}px',
        )
        print(
            "threshold      :",
            f'{report["used_threshold_px"]:.3f}px',
        )

        rejected = report[
            "rejected_indices"
        ]

        if not rejected:
            print(
                "rejected pairs : 0"
            )

            report["cleanup_performed"] = False

            save_report(
                report,
                report_path,
            )

        else:
            print(
                "rejected pairs :",
                len(rejected),
            )

            for pair in report["pairs"]:
                if pair["rejected"]:
                    print(
                        f'  {pair["left_frame"]} -> '
                        f'{pair["right_frame"]} '
                        f'score='
                        f'{pair["pair_score_px"]:.3f}px'
                    )

            remaining = (
                report["n_pairs"]
                - len(rejected)
            )

            if remaining < args.stable_min_pairs:
                print(
                    "Cleanup skipped: rejecting these pairs "
                    "would leave only",
                    remaining,
                    "pairs; minimum required is",
                    args.stable_min_pairs,
                )

                report[
                    "cleanup_performed"
                ] = False

                report[
                    "cleanup_skipped_reason"
                ] = (
                    "too_few_pairs_after_rejection"
                )

                save_report(
                    report,
                    report_path,
                )

            else:
                cleaned_inputs = (
                    output_root
                    / "inputs_outlier_cleaned"
                )

                shutil.rmtree(
                    cleaned_inputs,
                    ignore_errors=True,
                )

                cleaned_detection_paths = (
                    write_cleaned_detection_payloads(
                        detection_paths,
                        cleaned_inputs,
                        rejected,
                    )
                )

                cleaned_output = (
                    output_root
                    / "calibcam_output_cleaned"
                )

                shutil.rmtree(
                    cleaned_output,
                    ignore_errors=True,
                )

                cleaned_command = list(
                    command
                )

                detection_idx = (
                    cleaned_command.index(
                        "--detection"
                    )
                )

                cleaned_command[
                    detection_idx + 1
                ] = str(
                    cleaned_detection_paths[0]
                )

                cleaned_command[
                    detection_idx + 2
                ] = str(
                    cleaned_detection_paths[1]
                )

                data_idx = (
                    cleaned_command.index(
                        "--data_path"
                    )
                )

                cleaned_command[
                    data_idx + 1
                ] = str(
                    cleaned_output
                )

                print()
                print(
                    "Re-running CalibCam after "
                    "outlier rejection..."
                )

                subprocess.run(
                    cleaned_command,
                    check=True,
                )

                cleaned_positions = (
                    cleaned_output
                    / "multicam_calibration_board_positions.yml"
                )

                cleaned_calibration = (
                    cleaned_output
                    / "multicam_calibration.yml"
                )

                cleaned_analysis = (
                    analyze_board_positions(
                        cleaned_positions,
                        mad_multiplier=(
                            args.calibcam_outlier_mad_multiplier
                        ),
                        min_threshold_px=(
                            args.calibcam_outlier_min_px
                        ),
                        max_threshold_px=(
                            args.calibcam_outlier_max_px
                        ),
                    )
                )

                geometry = compare_extrinsics(
                    initial_calibration,
                    cleaned_calibration,
                )

                report[
                    "cleanup_performed"
                ] = True

                report[
                    "remaining_pairs"
                ] = remaining

                report[
                    "cleaned_detection_paths"
                ] = [
                    str(path)
                    for path
                    in cleaned_detection_paths
                ]

                report[
                    "cleaned_calibration_output"
                ] = str(
                    cleaned_output
                )

                report[
                    "cleaned_analysis"
                ] = cleaned_analysis

                report[
                    "geometry_comparison"
                ] = geometry

                save_report(
                    report,
                    report_path,
                )

                print()
                print(
                    "Cleaned Stage-2 calibration"
                )
                print(
                    "---------------------------"
                )
                print(
                    "remaining pairs:",
                    remaining,
                )
                print(
                    "baseline change:",
                    f'{geometry["baseline_change_mm"]:+.3f} mm',
                )
                print(
                    "rotation change:",
                    f'{geometry["rotation_change_deg"]:+.3f} deg',
                )
                print(
                    "cleaned output:",
                    cleaned_output,
                )
                print(
                    "outlier report:",
                    report_path,
                )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from nanodeepcharuco.pipeline.detection_runner import (
        build_and_save_payload, run_detector_on_video,
    )
    output_root = resolved(args.data_path)
    inputs_dir = output_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    detectors = (
        make_detector(args, "left", output_root),
        make_detector(args, "right", output_root),
    )
    if args.auto_sync:
        from nanodeepcharuco.pipeline.auto_sync import (
            run_auto_sync_stage2,
        )

        (
            detection_paths,
            synchronized_pairs,
            sync_segments,
            sync_windows,
        ) = run_auto_sync_stage2(
            args=args,
            detectors=detectors,
            output_root=output_root,
        )

        camera_data = [
            {
                pair.left_frame: {}
                for pair in synchronized_pairs
            },
            {
                pair.right_frame: {}
                for pair in synchronized_pairs
            },
        ]

    else:
        camera_data = []

        for side, video, detector, offset in zip(
            ("left", "right"),
            args.videos,
            detectors,
            args.frames_offsets,
        ):
            camera_data.append(
                run_detector_on_video(
                    video,
                    detector,
                    args.frames_step,
                    side,
                    frames_start=args.frames_start,
                    frames_end=args.frames_end,
                    frame_offset=offset,
                )
            )

        detection_paths = (
            inputs_dir / "detection_000.npy",
            inputs_dir / "detection_001.npy",
        )

        for data, path, offset in zip(
            camera_data,
            detection_paths,
            args.frames_offsets,
        ):
            build_and_save_payload(
                data,
                path,
                frame_offset=offset,
                frames_start=args.frames_start,
                frames_step=args.frames_step,
            )

    manifest = {
        "pipeline": "NanoDeepCharuco_Basic", "detector": args.detector,
        "videos": [str(path) for path in args.videos], "board": str(args.board),
        "frames_start": args.frames_start, "frames_end": args.frames_end,
        "frames_step": args.frames_step, "frames_offsets": list(args.frames_offsets),
        "models": list(args.models), "projection": args.projection,
        "calibration_mode": (
            "two_stage_intrinsics"
            if args.stage1_calibration
            else (
                "two_stage_extrinsics"
                if args.stage1_intrinsics is not None
                else "standard"
            )
        ),
        "stage1_intrinsics": (
            [str(path) for path in args.stage1_intrinsics]
            if args.stage1_intrinsics is not None
            else None
        ),
        "detections": [str(path) for path in detection_paths],
        "detected_frames": [len(data) for data in camera_data],
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Detection output: {inputs_dir}")
    if (
        args.run_calibcam
        or args.stage1_calibration
        or args.calibration_single
        or args.calibration_multi
        or args.stage1_intrinsics is not None
    ):
        run_calibcam(args, output_root, detection_paths)
        print(f"Calibration output: {output_root / 'calibcam_output'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
