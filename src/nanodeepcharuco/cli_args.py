from __future__ import annotations

import argparse
from pathlib import Path


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
        "--calibration_single",
        nargs="+",
        default=None,
        help=(
            "Existing single-camera CalibCam calibration YAMLs. "
            "With two videos, enables fixed-intrinsics, "
            "extrinsics-only calibration."
        ),
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

    calibration_single = getattr(
        args,
        "calibration_single",
        None,
    )

    if calibration_single is not None:
        if len(args.videos) != 2:
            raise ValueError(
                "--calibration_single currently requires "
                "exactly two videos."
            )

        if len(calibration_single) != len(args.videos):
            raise ValueError(
                "--calibration_single must contain one "
                "calibration file per video."
            )

    if args.models is not None:
        if calibration_single is not None:
            if len(args.models) not in (
                1,
                len(args.videos),
            ):
                raise ValueError(
                    "Two-stage calibration accepts either "
                    "one shared model or one model per video."
                )
        elif len(args.models) != len(args.videos):
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
