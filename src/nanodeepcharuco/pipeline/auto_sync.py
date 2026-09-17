from __future__ import annotations

from nanodeepcharuco.pipeline.detectors import (
    build_hybrid_detector,
)
from nanodeepcharuco.pipeline.sequence import (
    run_camera_mapped_sequence,
    remap_camera_run,
    restrict_camera_run,
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


def run_to_physical_detections(
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


def run_auto_sync_discovery(
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
        detector = build_hybrid_detector(
            config=config,
            deep_detector=deep,
            work_dir=(
                run_dir
                / "tmp"
                / f"autosync_camera_{cam_idx:03d}"
            ),
        )

        run = run_camera_mapped_sequence(
            video_path=video_path,
            detector=detector,
            physical_frames_by_index=(
                physical_map
            ),
            canonical_luma=(
                config.canonical_luma
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


def resolve_auto_sync_runs(
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
        run_to_physical_detections(
            left_discovery_run
        )
    )

    right_by_frame = (
        run_to_physical_detections(
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
