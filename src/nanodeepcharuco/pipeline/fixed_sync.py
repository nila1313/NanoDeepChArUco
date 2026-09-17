from __future__ import annotations

from pathlib import Path

from nanodeepcharuco.pipeline.detectors import (
    build_hybrid_detector,
)
from nanodeepcharuco.pipeline.sequence import (
    run_camera_sequence,
    restrict_camera_run,
)


def find_shared_detection_ids(
    camera_runs,
) -> list[int]:
    """Return logical detection IDs shared by all cameras."""

    if not camera_runs:
        return []

    return sorted(
        set.intersection(
            *[
                set(run.detections_by_index)
                for run in camera_runs
            ]
        )
    )


def run_fixed_sync_detection(
    *,
    config,
    logical_frame_ids,
    deep_detector,
    temp_dir: str | Path,
):
    """Run detection using the configured fixed frame offsets."""

    temp_dir = Path(temp_dir)
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
            deep_detector=deep_detector,
            work_dir=(
                temp_dir
                / f"camera_{cam_idx:03d}"
            ),
        )

        run = run_camera_sequence(
            video_path=video_path,
            detector=detector,
            logical_frame_ids=logical_frame_ids,
            frame_offset=frame_offset,
            canonical_luma=(
                config.canonical_luma
            ),
            camera_name=(
                f"camera_{cam_idx:03d}"
            ),
        )

        camera_runs.append(run)

    shared_detection_ids = (
        find_shared_detection_ids(
            camera_runs
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

    return camera_runs
