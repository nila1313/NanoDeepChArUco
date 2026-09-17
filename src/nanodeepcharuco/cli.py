from __future__ import annotations

import argparse
from pathlib import Path

from nanodeepcharuco.config import (
    apply_profile,
    apply_runtime_backend,
    resolve_config,
    save_resolved_config,
)

from nanodeepcharuco.run_layout import RunLayout
from nanodeepcharuco.calibcam.backend import (
    build_calibcam_command,
    run_calibcam,
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

from nanodeepcharuco.detection.charuco import (
    expected_charuco_corner_ids,
)

from nanodeepcharuco.pipeline.detectors import (
    build_hybrid_detector,
)

from nanodeepcharuco.pipeline.sequence import (
    run_camera_sequence,
    restrict_camera_run,
    save_camera_run_payload,
)

from nanodeepcharuco.pipeline.auto_sync import (
    run_auto_sync_discovery,
    resolve_auto_sync_runs,
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
        default=None,
        help=(
            "ChArUco board definition file. "
            "May also be supplied by --profile."
        ),
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
            "style as CalibCam. Supplying explicit offsets without "
            "--auto_sync selects fixed synchronization."
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
        "--profile",
        default=None,
        help=(
            "Portable NanoDeepChArUco asset profile name "
            "(for example large_7x7 or small_5x6), "
            "or path to a profile YAML file. "
            "Explicit asset path arguments override profile values."
        ),
    )

    parser.add_argument(
        "--nano_executable",
        default=None,
        help="Path to the ArUco Nano detect_batch executable.",
    )

    parser.add_argument(
        "--deepcharuco_root",
        default=None,
        help="Path to the DeepChArUco source directory.",
    )

    parser.add_argument(
        "--deep_checkpoint",
        default=None,
        help="Path to the DeepChArUco detector checkpoint.",
    )

    parser.add_argument(
        "--refinenet_checkpoint",
        default=None,
        help="Path to the DeepChArUco RefineNet checkpoint.",
    )

    parser.add_argument(
        "--deep_config",
        default=None,
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

    luma_group = parser.add_mutually_exclusive_group()

    luma_group.add_argument(
        "--canonical_luma",
        dest="canonical_luma",
        action="store_true",
        help="Use canonical native-luminance decoding for cross-platform reproducibility. This is the default.",
    )

    luma_group.add_argument(
        "--no_canonical_luma",
        dest="canonical_luma",
        action="store_false",
        help="Use legacy platform-dependent BGR video decoding instead of canonical luminance.",
    )

    parser.set_defaults(canonical_luma=True)

    parser.add_argument(
        "--auto_sync",
        action="store_true",
        help=(
            "Explicitly enable automatic piecewise stereo "
            "synchronization. This is the default for a two-video "
            "run when --frames_offsets is omitted."
        ),
    )

    parser.add_argument(
        "--fixed_sync",
        action="store_true",
        help=(
            "Use fixed per-camera synchronization instead of "
            "automatic synchronization. Frame offsets are taken "
            "from --frames_offsets; omitted offsets default to zero."
        ),
    )

    parser.add_argument(
        "--sync_offsets",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Candidate right-camera offsets for automatic "
            "synchronization. Default: -3 -2 -1 0 1 2 3."
        ),
    )

    parser.add_argument(
        "--sync_frame_step",
        type=int,
        default=1,
        help=(
            "Physical frame sampling step used only for "
            "automatic synchronization discovery. "
            "Independent from --frames_step."
        ),
    )

    parser.add_argument(
        "--sync_window_size",
        type=int,
        default=100,
        help="Automatic-sync window size in frames.",
    )

    parser.add_argument(
        "--sync_window_step",
        type=int,
        default=100,
        help="Automatic-sync window step in frames.",
    )

    parser.add_argument(
        "--sync_min_inlier_ratio",
        type=float,
        default=0.20,
        help=(
            "Minimum epipolar inlier ratio for a "
            "trusted synchronization window."
        ),
    )

    parser.add_argument(
        "--sync_min_ratio_gap",
        type=float,
        default=0.05,
        help=(
            "Minimum inlier-ratio advantage over the "
            "second-best offset."
        ),
    )

    parser.add_argument(
        "--sync_min_persistence",
        type=int,
        default=2,
        help=(
            "Minimum number of supporting windows for "
            "a stable synchronization segment."
        ),
    )

    parser.add_argument(
        "--sync_max_gap_frames",
        type=int,
        default=400,
        help=(
            "Maximum no-evidence gap allowed between "
            "supporting windows of the same offset."
        ),
    )

    parser.add_argument(
        "--calibcam_python",
        default=None,
        help=(
            "Optional Python executable override for CalibCam. "
            "If omitted, NanoDeepChArUco uses the runtime backend "
            "configured by setup.sh or "
            "NANODEEPCHARUCO_CALIBCAM_PYTHON."
        ),
    )

    return parser


def resolve_sync_mode(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """
    Resolve the user-facing synchronization mode.

    Two-video runs default to automatic synchronization when no
    explicit frame offsets or synchronization mode are supplied.

    Explicit frame offsets preserve the historical fixed-offset
    behavior unless --auto_sync is requested.
    """
    if args.auto_sync and args.fixed_sync:
        raise ValueError(
            "--auto_sync and --fixed_sync cannot be used together."
        )

    if args.fixed_sync:
        args.auto_sync = False

    elif not args.auto_sync:
        args.auto_sync = (
            len(args.videos) == 2
            and args.frames_offsets is None
        )

    if (
        args.auto_sync
        and args.frames_offsets is None
    ):
        args.frames_offsets = [
            0
            for _ in args.videos
        ]

    return args


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

    if args.auto_sync:
        if len(args.videos) != 2:
            raise ValueError(
                "--auto_sync currently requires exactly "
                "two videos."
            )

        auto_sync_offsets = (
            args.frames_offsets
            if args.frames_offsets is not None
            else [0] * len(args.videos)
        )

        if any(
            int(offset) != 0
            for offset in auto_sync_offsets
        ):
            raise ValueError(
                "--auto_sync currently requires "
                "--frames_offsets 0 0. "
                "Temporal offsets are estimated automatically."
            )

        if args.sync_frame_step <= 0:
            raise ValueError(
                "--sync_frame_step must be greater than zero."
            )

        if args.sync_window_size <= 0:
            raise ValueError(
                "--sync_window_size must be greater than zero."
            )

        if args.sync_window_step <= 0:
            raise ValueError(
                "--sync_window_step must be greater than zero."
            )

        if args.sync_min_persistence <= 0:
            raise ValueError(
                "--sync_min_persistence must be greater than zero."
            )

        if args.sync_max_gap_frames < 0:
            raise ValueError(
                "--sync_max_gap_frames cannot be negative."
            )

        if (
            args.sync_offsets is not None
            and len(args.sync_offsets) == 0
        ):
            raise ValueError(
                "--sync_offsets must contain at least one offset."
            )

    for video in args.videos:
        if not Path(video).is_file():
            raise FileNotFoundError(
                f"Video not found: {video}"
            )

    if args.board is None:
        raise ValueError(
            "Board file is required. "
            "Provide --board or use a profile "
            "that defines a board."
        )

    if not Path(args.board).is_file():
        raise FileNotFoundError(
            f"Board file not found: {args.board}"
        )

    required_assets = {
        "Nano executable": args.nano_executable,
        "DeepChArUco source": args.deepcharuco_root,
        "Deep checkpoint": args.deep_checkpoint,
        "RefineNet checkpoint": args.refinenet_checkpoint,
        "Deep config": args.deep_config,
    }

    missing_assets = [
        name
        for name, value
        in required_assets.items()
        if value is None
    ]

    if missing_assets:
        raise ValueError(
            "Missing NanoDeepChArUco assets: "
            + ", ".join(missing_assets)
            + ". Provide --profile or explicit path arguments."
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

    args = apply_profile(args)
    args = apply_runtime_backend(args)
    args = resolve_sync_mode(args)

    validate_args(args)

    config = resolve_config(args)

    layout = RunLayout.from_data_path(
        config.data_path
    )
    layout.prepare()

    config_path = save_resolved_config(
        config,
        layout.metadata_dir,
    )

    manifest, _ = build_input_manifest(
        config
    )

    manifest_path = save_input_manifest(
        manifest,
        layout.metadata_dir,
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

    expected_marker_ids = (
        expected_charuco_corner_ids(
            config.board
        )
    )

    if config.auto_sync:
        print()
        print(
            "auto_sync     : enabled"
        )

        (
            left_discovery_run,
            right_discovery_run,
        ) = run_auto_sync_discovery(
            config=config,
            run_dir=layout.metadata_dir,
            video_infos=video_infos,
            deep=deep,
        )

        (
            left_run,
            right_run,
            _segments,
            _window_results,
        ) = resolve_auto_sync_runs(
            config=config,
            run_dir=layout.metadata_dir,
            calibration_frame_ids=logical_ids,
            left_discovery_run=(
                left_discovery_run
            ),
            right_discovery_run=(
                right_discovery_run
            ),
        )

        camera_runs = [
            left_run,
            right_run,
        ]

    else:
        print()
        print(
            "auto_sync     : disabled"
        )

        camera_runs = []

        for cam_idx, (
            video_path,
            frame_offset,
        ) in enumerate(
            zip(
                config.videos,
                config.frames_offsets,
            )
        ):
            detector = build_hybrid_detector(
                config=config,
                deep_detector=deep,
                work_dir=(
                    layout.temp_dir
                    / f"camera_{cam_idx:03d}"
                ),
            )

            run = run_camera_sequence(
                video_path=video_path,
                detector=detector,
                logical_frame_ids=logical_ids,
                frame_offset=frame_offset,
                canonical_luma=(
                    config.canonical_luma
                ),
                camera_name=(
                    f"camera_{cam_idx:03d}"
                ),
            )

            camera_runs.append(
                run
            )

        shared_detection_ids = sorted(
            set.intersection(
                *[
                    set(
                        run.detections_by_index
                    )
                    for run in camera_runs
                ]
            )
        )

        if not shared_detection_ids:
            raise RuntimeError(
                "No shared logical detection indices "
                "remain across cameras."
            )

        camera_runs = [
            restrict_camera_run(
                run,
                shared_detection_ids,
            )
            for run in camera_runs
        ]

        print()
        print(
            "shared fixed-offset stereo frames:",
            len(shared_detection_ids),
        )
        print(
            "shared logical range             :",
            shared_detection_ids[0],
            "to",
            shared_detection_ids[-1],
        )

    for cam_idx, run in enumerate(
        camera_runs
    ):
        output_path = (
            layout.detection_path(
                cam_idx
            )
        )

        payload = save_camera_run_payload(
            run,
            output_path,
            expected_marker_ids=(
                expected_marker_ids
            ),
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
        detection_paths = [
            layout.detection_path(
                cam_idx
            )
            for cam_idx
            in range(len(config.videos))
        ]

        cmd = build_calibcam_command(
            python_executable=config.calibcam_python,
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
                for part in cmd
            )
        )
        print()

        run_calibcam(
            cmd
        )

        print()
        print("=" * 80)
        print("CALIBCAM BACKEND FINISHED")
        print("=" * 80)
        print(
            "Calibration output:",
            layout.calibcam_data_path,
        )


if __name__ == "__main__":
    main()
