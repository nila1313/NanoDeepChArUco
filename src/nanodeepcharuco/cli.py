from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from nanodeepcharuco.config import (
    apply_profile,
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
    run_camera_mapped_sequence,
    remap_camera_run,
    restrict_camera_run,
    save_camera_run_payload,
)

from nanodeepcharuco.sync.schedule import (
    build_auto_sync_detection_maps,
)

from nanodeepcharuco.sync.windowed import (
    search_windowed_offsets,
)

from nanodeepcharuco.sync.segments import (
    build_sync_segments,
)

from nanodeepcharuco.sync.pairing import (
    build_synchronized_pairs,
)

from nanodeepcharuco.sync.reporting import (
    save_sync_reports,
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

    parser.add_argument(
        "--auto_sync",
        action="store_true",
        help=(
            "Automatically estimate piecewise stereo "
            "frame synchronization."
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

    if args.auto_sync:
        if len(args.videos) != 2:
            raise ValueError(
                "--auto_sync currently requires exactly "
                "two videos."
            )

        if any(
            int(offset) != 0
            for offset in args.frames_offsets
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


def _run_to_physical_detections(
    run,
) -> dict[int, dict]:
    """
    Convert a CameraDetectionRun into:

        physical_frame -> detected corners

    Automatic synchronization operates on physical
    video-frame numbers rather than CalibCam detection IDs.
    """

    detections = {}

    for detection_idx, corners in (
        run.detections_by_index.items()
    ):
        physical_idx = (
            run.frame_indices_by_index.get(
                detection_idx
            )
        )

        if physical_idx is None:
            continue

        detections[
            int(physical_idx)
        ] = corners

    return detections


def _run_auto_sync_discovery(
    config,
    run_dir,
    video_infos,
    deep,
):
    """
    Run the dense/padded detection pass required for
    automatic stereo synchronization.

    The left camera defines the reference timeline.
    The right camera additionally detects every physical
    frame needed by the candidate synchronization offsets.
    """

    if config.frames_end is None:
        reference_end = min(
            int(video_infos[0].frame_count),
            int(video_infos[1].frame_count),
        )
    else:
        reference_end = min(
            int(config.frames_end),
            int(video_infos[0].frame_count),
            int(video_infos[1].frame_count),
        )

    left_map, right_map = (
        build_auto_sync_detection_maps(
            frames_start=config.frames_start,
            frames_end=reference_end,
            frames_step=config.sync_frame_step,
            offsets=config.sync_offsets,
            left_frame_count=(
                video_infos[0].frame_count
            ),
            right_frame_count=(
                video_infos[1].frame_count
            ),
        )
    )

    print()
    print("Automatic synchronization discovery")
    print("-----------------------------------")
    print(
        "candidate offsets:",
        config.sync_offsets,
    )
    print(
        "sync frame step  :",
        config.sync_frame_step,
    )
    print(
        "left discovery frames :",
        len(left_map),
    )
    print(
        "right discovery frames:",
        len(right_map),
    )

    runs = []

    for cam_idx, (
        video_path,
        physical_map,
    ) in enumerate(
        [
            (
                config.videos[0],
                left_map,
            ),
            (
                config.videos[1],
                right_map,
            ),
        ]
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
                / f"autosync_camera_{cam_idx:03d}"
            ),
            gamma=config.gamma,
            deep_self_ransac_px=(
                config.deep_self_ransac_px
            ),
        )

        run = run_camera_mapped_sequence(
            video_path=video_path,
            detector=detector,
            physical_frames_by_index=(
                physical_map
            ),
            camera_name=(
                f"camera_{cam_idx:03d}"
            ),
        )

        runs.append(run)

    return (
        runs[0],
        runs[1],
    )


def _resolve_auto_sync_runs(
    config,
    run_dir,
    calibration_frame_ids,
    left_discovery_run,
    right_discovery_run,
):
    """
    Estimate stable piecewise stereo synchronization and
    remap the already-computed discovery detections.

    Returns
    -------
    left_run
        Left detections indexed by synchronized stereo ID.

    right_run
        Right detections indexed by the same stereo ID.

    segments
        Stable piecewise synchronization segments.

    window_results
        Raw per-window synchronization measurements.
    """

    left_by_frame = (
        _run_to_physical_detections(
            left_discovery_run
        )
    )

    right_by_frame = (
        _run_to_physical_detections(
            right_discovery_run
        )
    )

    print()
    print("Automatic synchronization scoring")
    print("---------------------------------")

    window_results = (
        search_windowed_offsets(
            left_by_frame=left_by_frame,
            right_by_frame=right_by_frame,
            offsets=config.sync_offsets,
            window_size=(
                config.sync_window_size
            ),
            window_step=(
                config.sync_window_step
            ),
            min_shared_per_frame=6,
            ransac_threshold_px=1.5,
            min_frame_pairs=4,
        )
    )

    for result in window_results:
        if result.best_score is None:
            print(
                f"{result.start_frame}-"
                f"{result.end_frame - 1}: "
                "NONE"
            )
            continue

        print(
            f"{result.start_frame}-"
            f"{result.end_frame - 1}: "
            f"offset="
            f"{result.best_offset:+d} "
            f"ratio="
            f"{result.best_score.inlier_ratio:.4f} "
            f"gap="
            f"{result.inlier_ratio_gap:.4f} "
            f"pairs="
            f"{result.best_score.n_frame_pairs}"
        )

    segments = build_sync_segments(
        window_results,
        min_inlier_ratio=(
            config.sync_min_inlier_ratio
        ),
        min_ratio_gap=(
            config.sync_min_ratio_gap
        ),
        min_persistence=(
            config.sync_min_persistence
        ),
        max_gap_frames=(
            config.sync_max_gap_frames
        ),
    )

    print()
    print("Stable synchronization segments")
    print("--------------------------------")

    if not segments:
        raise RuntimeError(
            "Automatic synchronization could not "
            "find any stable synchronization segments."
        )

    for segment in segments:
        print(
            f"{segment.start_frame}-"
            f"{segment.end_frame - 1}: "
            f"offset={segment.offset:+d} "
            f"support="
            f"{segment.n_support_windows} "
            f"mean_ratio="
            f"{segment.mean_inlier_ratio:.4f} "
            f"mean_gap="
            f"{segment.mean_ratio_gap:.4f}"
        )

    left_frames = [
        int(frame_idx)
        for frame_idx in calibration_frame_ids
        if int(frame_idx) in left_by_frame
    ]

    right_frames = sorted(
        right_by_frame.keys()
    )

    print()
    print("Final calibration sampling")
    print("--------------------------")
    print(
        "requested calibration frames:",
        len(calibration_frame_ids),
    )
    print(
        "available calibration frames:",
        len(left_frames),
    )
    print(
        "pairing policy             : "
        "trusted support windows only"
    )

    synchronized_pairs = (
        build_synchronized_pairs(
            left_frames=left_frames,
            right_frames=right_frames,
            segments=segments,
            require_right_available=True,
            window_results=window_results,
            min_inlier_ratio=(
                config.sync_min_inlier_ratio
            ),
            min_ratio_gap=(
                config.sync_min_ratio_gap
            ),
        )
    )

    if not synchronized_pairs:
        raise RuntimeError(
            "Stable synchronization segments were found, "
            "but no usable synchronized detection pairs "
            "could be constructed."
        )

    left_map = {
        pair.left_frame:
        pair.left_frame
        for pair in synchronized_pairs
    }

    right_map = {
        pair.left_frame:
        pair.right_frame
        for pair in synchronized_pairs
    }

    left_run = remap_camera_run(
        run=left_discovery_run,
        physical_frames_by_index=(
            left_map
        ),
    )

    right_run = remap_camera_run(
        run=right_discovery_run,
        physical_frames_by_index=(
            right_map
        ),
    )

    shared_detection_ids = sorted(
        set(
            left_run.detections_by_index
        )
        & set(
            right_run.detections_by_index
        )
    )

    if not shared_detection_ids:
        raise RuntimeError(
            "Automatic synchronization produced no "
            "shared stereo detections."
        )

    # Final stereo payloads must contain exactly the same
    # logical detection IDs on both cameras.
    left_run = restrict_camera_run(
        left_run,
        shared_detection_ids,
    )

    right_run = restrict_camera_run(
        right_run,
        shared_detection_ids,
    )

    shared_detection_id_set = set(
        shared_detection_ids
    )

    synchronized_pairs = [
        pair
        for pair in synchronized_pairs
        if pair.left_frame
        in shared_detection_id_set
    ]

    print()
    print(
        "synchronized stereo pairs:",
        len(shared_detection_ids),
    )

    print(
        "synchronized range       :",
        shared_detection_ids[0],
        "to",
        shared_detection_ids[-1],
    )

    (
        window_report_path,
        segment_report_path,
        pair_report_path,
    ) = save_sync_reports(
        run_dir=run_dir,
        window_results=window_results,
        segments=segments,
        synchronized_pairs=(
            synchronized_pairs
        ),
    )

    print()
    print("Synchronization reports")
    print("-----------------------")
    print(
        "windows :",
        window_report_path,
    )
    print(
        "segments:",
        segment_report_path,
    )
    print(
        "pairs   :",
        pair_report_path,
    )

    return (
        left_run,
        right_run,
        segments,
        window_results,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args = apply_profile(args)

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

    if config.auto_sync:
        print()
        print(
            "auto_sync     : enabled"
        )

        (
            left_discovery_run,
            right_discovery_run,
        ) = _run_auto_sync_discovery(
            config=config,
            run_dir=run_dir,
            video_infos=video_infos,
            deep=deep,
        )

        (
            left_run,
            right_run,
            _segments,
            _window_results,
        ) = _resolve_auto_sync_runs(
            config=config,
            run_dir=run_dir,
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

        for cam_idx, run in enumerate(
            camera_runs
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
                    / f"payload_camera_{cam_idx:03d}"
                ),
                gamma=config.gamma,
                deep_self_ransac_px=(
                    config.deep_self_ransac_px
                ),
            )

            output_path = (
                run_dir
                / f"detection_{cam_idx:03d}.yml"
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

    else:
        print()
        print(
            "auto_sync     : disabled"
        )

        camera_runs = []
        expected_marker_ids_by_camera = []

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

            camera_runs.append(
                run
            )

            expected_n_ids = (
                (detector.board_width - 1)
                * (detector.board_height - 1)
            )

            expected_marker_ids_by_camera.append(
                list(
                    range(expected_n_ids)
                )
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
                run_dir
                / f"detection_{cam_idx:03d}.yml"
            )

            payload = save_camera_run_payload(
                run,
                output_path,
                expected_marker_ids=(
                    expected_marker_ids_by_camera[
                        cam_idx
                    ]
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
            / f"detection_{cam_idx:03d}.yml"
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
