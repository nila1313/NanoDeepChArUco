from __future__ import annotations

from pathlib import Path

import cv2

from nanodeepcharuco.calibcam.payload import (
    build_calibcam_dict_from_selected_frames,
    save_calibcam_detection,
)
from nanodeepcharuco.pipeline.detection_runner import (
    run_detector_on_frames,
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


MIN_FINAL_SHARED_CORNERS = 5


def _video_frame_count(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    try:
        return int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
    finally:
        cap.release()


def _filter_usable_pairs(
    pairs,
    left_by_frame,
    right_by_frame,
    min_shared_corners: int = MIN_FINAL_SHARED_CORNERS,
):
    usable = []

    for pair in pairs:
        left = left_by_frame.get(
            pair.left_frame,
            {},
        )

        right = right_by_frame.get(
            pair.right_frame,
            {},
        )

        shared = (
            set(left)
            & set(right)
        )

        if len(shared) < min_shared_corners:
            continue

        usable.append(pair)

    return usable


def run_auto_sync_stage2(
    args,
    detectors,
    output_root: Path,
):
    left_count = _video_frame_count(
        args.videos[0]
    )

    right_count = _video_frame_count(
        args.videos[1]
    )

    reference_end = min(
        left_count,
        right_count,
    )

    if args.frames_end is not None:
        reference_end = min(
            reference_end,
            int(args.frames_end),
        )

    left_map, right_map = (
        build_auto_sync_detection_maps(
            frames_start=args.frames_start,
            frames_end=reference_end,
            frames_step=args.sync_frame_step,
            offsets=list(args.sync_offsets),
            left_frame_count=left_count,
            right_frame_count=right_count,
        )
    )

    print()
    print("Automatic synchronization discovery")
    print("-----------------------------------")
    print(
        "candidate offsets:",
        list(args.sync_offsets),
    )
    print(
        "left discovery frames :",
        len(left_map),
    )
    print(
        "right discovery frames:",
        len(right_map),
    )

    left_by_frame = run_detector_on_frames(
        args.videos[0],
        detectors[0],
        left_map.keys(),
        "left_sync",
    )

    right_by_frame = run_detector_on_frames(
        args.videos[1],
        detectors[1],
        right_map.keys(),
        "right_sync",
    )

    window_results = search_windowed_offsets(
        left_by_frame=left_by_frame,
        right_by_frame=right_by_frame,
        offsets=list(args.sync_offsets),
        window_size=args.sync_window_size,
        window_step=args.sync_window_step,
        min_shared_per_frame=6,
        ransac_threshold_px=1.5,
        min_frame_pairs=4,
    )

    print()
    print("Synchronization windows")
    print("-----------------------")

    for result in window_results:
        if result.best_score is None:
            print(
                f"{result.start_frame}-"
                f"{result.end_frame - 1}: "
                "no trusted candidate"
            )
            continue

        print(
            f"{result.start_frame}-"
            f"{result.end_frame - 1}: "
            f"offset={result.best_offset:+d} "
            f"inlier_ratio="
            f"{result.best_score.inlier_ratio:.4f} "
            f"gap="
            f"{result.inlier_ratio_gap:.4f}"
        )

    segments = build_sync_segments(
        window_results,
        min_inlier_ratio=args.sync_min_inlier_ratio,
        min_ratio_gap=args.sync_min_ratio_gap,
        min_persistence=args.sync_min_persistence,
        max_gap_frames=args.sync_max_gap_frames,
    )

    if not segments:
        raise RuntimeError(
            "Automatic synchronization found no "
            "trusted stable synchronization segments."
        )

    print()
    print("Trusted synchronization segments")
    print("--------------------------------")

    for segment in segments:
        print(
            f"{segment.start_frame}-"
            f"{segment.end_frame - 1}: "
            f"offset={segment.offset:+d} "
            f"support={segment.n_support_windows}"
        )

    calibration_left_frames = [
        frame
        for frame in range(
            int(args.frames_start),
            int(reference_end),
            int(args.frames_step),
        )
        if frame in left_by_frame
    ]

    pairs = build_synchronized_pairs(
        left_frames=calibration_left_frames,
        right_frames=sorted(right_by_frame),
        segments=segments,
        require_right_available=True,
        window_results=window_results,
        min_inlier_ratio=args.sync_min_inlier_ratio,
        min_ratio_gap=args.sync_min_ratio_gap,
    )

    pairs = _filter_usable_pairs(
        pairs,
        left_by_frame,
        right_by_frame,
    )

    if not pairs:
        raise RuntimeError(
            "Synchronization succeeded, but no "
            "usable stereo pairs remained."
        )

    print()
    print("Final synchronized Stage-2 sampling")
    print("-----------------------------------")
    print(
        "usable stereo pairs:",
        len(pairs),
    )

    inputs_dir = output_root / "inputs"

    inputs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    left_path = (
        inputs_dir
        / "detection_000.npy"
    )

    right_path = (
        inputs_dir
        / "detection_001.npy"
    )

    detection_ids = list(
        range(len(pairs))
    )

    left_frames = [
        pair.left_frame
        for pair in pairs
    ]

    right_frames = [
        pair.right_frame
        for pair in pairs
    ]

    left_payload = (
        build_calibcam_dict_from_selected_frames(
            left_by_frame,
            left_frames,
            detection_ids,
        )
    )

    right_payload = (
        build_calibcam_dict_from_selected_frames(
            right_by_frame,
            right_frames,
            detection_ids,
        )
    )

    save_calibcam_detection(
        left_path,
        left_payload,
    )

    save_calibcam_detection(
        right_path,
        right_payload,
    )

    save_sync_reports(
        run_dir=output_root,
        window_results=window_results,
        segments=segments,
        synchronized_pairs=pairs,
    )

    return (
        (left_path, right_path),
        pairs,
        segments,
        window_results,
    )
