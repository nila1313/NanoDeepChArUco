from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from nanodeepcharuco.config import (
    prepare_run_directory,
    resolve_config,
    save_resolved_config,
)
from nanodeepcharuco.manifest import (
    build_input_manifest,
    save_input_manifest,
)

from nanodeepcharuco.video import (
    inspect_videos,
    build_logical_frame_ids,
)

from nanodeepcharuco.detection.deep import (
    DeepCharucoDetector,
)

from nanodeepcharuco.detection.hybrid import (
    NanoDeepCharucoDetector,
)

from nanodeepcharuco.pipeline.sequence import (
    run_camera_sequence,
    save_camera_run_payload,
)

from nanodeepcharuco.calibcam_io import (
    summarize_calibcam_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanodeepcharuco",
        description=(
            "NanoDeepChArUco detection and calibration pipeline "
            "with a CalibCam-style command-line interface."
        ),
    )

    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Input calibration videos.",
    )

    parser.add_argument(
        "--board",
        required=True,
        help="ChArUco board definition file.",
    )

    parser.add_argument(
        "--frames_start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--frames_end",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--frames_step",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--frames_offsets",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Per-camera frame offsets using the same user-facing "
            "style as CalibCam."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["pinhole", "omnidir"],
        default=None,
    )

    parser.add_argument(
        "--projection",
        choices=[
            "perspective",
            "fisheye_equidistant",
        ],
        default="perspective",
    )

    parser.add_argument(
        "--data_path",
        required=True,
        help="Output directory for detections and calibration results.",
    )

    parser.add_argument(
        "--detect_only",
        action="store_true",
        help="Run NanoDeepChArUco detection but skip CalibCam calibration.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.4,
        help="Gamma value used for the Nano retry stage.",
    )

    parser.add_argument(
        "--deep_self_ransac_px",
        type=float,
        default=5.0,
        help="Deep self-verification RANSAC threshold in pixels.",
    )

    parser.add_argument(
        "--nano_executable",
        required=True,
        help="Path to the ArUco Nano detect_batch executable.",
    )

    parser.add_argument(
        "--deepcharuco_root",
        required=True,
        help="Path to the DeepChArUco source directory.",
    )

    parser.add_argument(
        "--deep_checkpoint",
        required=True,
        help="Path to the DeepChArUco detector checkpoint.",
    )

    parser.add_argument(
        "--refinenet_checkpoint",
        required=True,
        help="Path to the DeepChArUco RefineNet checkpoint.",
    )

    parser.add_argument(
        "--deep_config",
        required=True,
        help="Path to the DeepChArUco YAML configuration.",
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
            "mps",
        ],
        default=None,
        help=(
            "Optional DeepChArUco device override. "
            "If omitted, the detector selects MPS, CUDA, or CPU automatically."
        ),
    )

    parser.add_argument(
        "--calibcam_python",
        default=None,
        help=(
            "Python executable for the verified CalibCam environment. "
            "Required unless --detect_only is used."
        ),
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if len(args.videos) < 1:
        raise ValueError("At least one input video is required.")

    if args.frames_step <= 0:
        raise ValueError("--frames_step must be greater than zero.")

    if args.frames_offsets is not None:
        if len(args.frames_offsets) != len(args.videos):
            raise ValueError(
                "--frames_offsets must contain one value per video."
            )

    if args.models is not None:
        if len(args.models) != len(args.videos):
            raise ValueError(
                "--models must contain one value per video."
            )

    for video in args.videos:
        if not Path(video).is_file():
            raise FileNotFoundError(
                f"Video not found: {video}"
            )

    if not Path(args.board).is_file():
        raise FileNotFoundError(
            f"Board file not found: {args.board}"
        )

    if not Path(args.nano_executable).is_file():
        raise FileNotFoundError(
            "Nano executable not found: "
            f"{args.nano_executable}"
        )

    if not Path(args.deepcharuco_root).is_dir():
        raise NotADirectoryError(
            "DeepChArUco source directory not found: "
            f"{args.deepcharuco_root}"
        )

    for name, value in [
        ("Deep checkpoint", args.deep_checkpoint),
        ("RefineNet checkpoint", args.refinenet_checkpoint),
        ("Deep config", args.deep_config),
    ]:
        if not Path(value).is_file():
            raise FileNotFoundError(
                f"{name} not found: {value}"
            )

    if not args.detect_only:
        if args.calibcam_python is None:
            raise ValueError(
                "--calibcam_python is required unless "
                "--detect_only is used."
            )

        if not Path(args.calibcam_python).is_file():
            raise FileNotFoundError(
                "CalibCam Python executable not found: "
                f"{args.calibcam_python}"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_args(args)

    config = resolve_config(args)
    run_dir = prepare_run_directory(config)
    config_path = save_resolved_config(
        config,
        run_dir,
    )

    manifest, _ = build_input_manifest(
        config
    )

    manifest_path = save_input_manifest(
        manifest,
        run_dir,
    )

    print("NanoDeepChArUco")
    print("================")
    print("videos        :", config.videos)
    print("board         :", config.board)
    print("frames_start  :", config.frames_start)
    print("frames_end    :", config.frames_end)
    print("frames_step   :", config.frames_step)
    print("frames_offsets:", config.frames_offsets)
    print("models        :", config.models)
    print("projection    :", config.projection)
    print("data_path     :", config.data_path)
    print("detect_only   :", config.detect_only)
    print("config saved  :", config_path)
    print("manifest saved:", manifest_path)

    video_infos = inspect_videos(
        config.videos
    )

    logical_ids = build_logical_frame_ids(
        video_infos=video_infos,
        frames_start=config.frames_start,
        frames_end=config.frames_end,
        frames_step=config.frames_step,
        frames_offsets=config.frames_offsets,
    )

    print()
    print(
        "logical frames :",
        len(logical_ids),
    )

    if len(logical_ids) == 0:
        raise RuntimeError(
            "No logical frames were selected."
        )

    print(
        "logical range  :",
        int(logical_ids[0]),
        "to",
        int(logical_ids[-1]),
    )

    deep = DeepCharucoDetector(
        deepcharuco_root=(
            config.deepcharuco_root
        ),
        deep_checkpoint=(
            config.deep_checkpoint
        ),
        refinenet_checkpoint=(
            config.refinenet_checkpoint
        ),
        config_path=(
            config.deep_config
        ),
        device=config.device,
    ).load()

    payloads = []

    for cam_idx, (
        video_path,
        frame_offset,
    ) in enumerate(
        zip(
            config.videos,
            config.frames_offsets,
        )
    ):
        detector = NanoDeepCharucoDetector(
            board_path=config.board,
            nano_executable=(
                config.nano_executable
            ),
            deep_detector=deep,
            work_dir=(
                run_dir
                / "tmp"
                / f"camera_{cam_idx:03d}"
            ),
            gamma=config.gamma,
            deep_self_ransac_px=(
                config.deep_self_ransac_px
            ),
        )

        run = run_camera_sequence(
            video_path=video_path,
            detector=detector,
            logical_frame_ids=logical_ids,
            frame_offset=frame_offset,
            camera_name=(
                f"camera_{cam_idx:03d}"
            ),
        )

        output_path = (
            run_dir
            / f"detection_{cam_idx:03d}.npy"
        )

        expected_n_ids = (
            (detector.board_width - 1)
            * (detector.board_height - 1)
        )

        expected_marker_ids = list(
            range(expected_n_ids)
        )

        payload = save_camera_run_payload(
            run,
            output_path,
            expected_marker_ids=(
                expected_marker_ids
            ),
        )

        payloads.append(
            payload
        )

        print()
        print(
            f"Camera {cam_idx} payload:"
        )
        print(
            summarize_calibcam_payload(
                payload
            )
        )
        print(
            "saved:",
            output_path,
        )

    print()
    print("Detection finished.")

    if config.detect_only:
        print(
            "detect_only=True; "
            "CalibCam calibration skipped."
        )

    else:
        calibcam_output = (
            run_dir
            / "calibcam_output"
        )

        calibcam_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        detection_paths = [
            run_dir
            / f"detection_{cam_idx:03d}.npy"
            for cam_idx
            in range(len(config.videos))
        ]

        cmd = [
            config.calibcam_python,
            "-m",
            "calibcam",
            "--videos",
            *config.videos,
            "--detection",
            *[
                str(path)
                for path in detection_paths
            ],
            "--board",
            config.board,
            "--calibration_single",
            "--calibration_multi",
            "--models",
            *config.models,
            "--projection",
            config.projection,
            "--data_path",
            str(calibcam_output),
        ]

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
            calibcam_output,
        )

        print()
        print("Command:")
        print(
            " ".join(
                str(part)
                for part in cmd
            )
        )
        print()

        subprocess.run(
            cmd,
            check=True,
        )

        print()
        print("=" * 80)
        print("CALIBCAM BACKEND FINISHED")
        print("=" * 80)
        print(
            "Calibration output:",
            calibcam_output,
        )


if __name__ == "__main__":
    main()
