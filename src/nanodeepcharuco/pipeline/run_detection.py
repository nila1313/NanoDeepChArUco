from __future__ import annotations

from pathlib import Path

from nanodeepcharuco.calibcam_io import (
    summarize_calibcam_payload,
)
from nanodeepcharuco.detection.charuco import (
    expected_charuco_corner_ids,
)
from nanodeepcharuco.detection.deep import (
    DeepCharucoDetector,
)
from nanodeepcharuco.pipeline.auto_sync import (
    resolve_auto_sync_runs,
    run_auto_sync_discovery,
)
from nanodeepcharuco.pipeline.fixed_sync import (
    run_fixed_sync_detection,
)
from nanodeepcharuco.pipeline.sequence import (
    save_camera_run_payload,
)
from nanodeepcharuco.run_layout import RunLayout


def load_deep_detector(config):
    """Load the configured DeepChArUco detector."""

    return DeepCharucoDetector(
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


def run_detection_pipeline(
    *,
    config,
    layout: RunLayout,
    video_infos,
    logical_frame_ids,
) -> list[Path]:
    """Run detection and write CalibCam-compatible payloads."""

    deep_detector = load_deep_detector(
        config
    )

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
            deep=deep_detector,
        )

        (
            left_run,
            right_run,
            _segments,
            _window_results,
        ) = resolve_auto_sync_runs(
            config=config,
            run_dir=layout.metadata_dir,
            calibration_frame_ids=(
                logical_frame_ids
            ),
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

        camera_runs = (
            run_fixed_sync_detection(
                config=config,
                logical_frame_ids=(
                    logical_frame_ids
                ),
                deep_detector=deep_detector,
                temp_dir=layout.temp_dir,
            )
        )

    detection_paths = []

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

        detection_paths.append(
            output_path
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

    return detection_paths
