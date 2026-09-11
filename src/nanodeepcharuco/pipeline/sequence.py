from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from nanodeepcharuco.calibcam_io import (
    build_calibcam_payload,
    save_calibcam_payload,
)
from nanodeepcharuco.video import (
    physical_frame_index,
)


@dataclass
class FrameAudit:
    detection_idx: int
    physical_frame_idx: int
    status: str
    corner_count: int
    nano_source: str
    nano_base: int
    deep_ran: bool
    deep_raw: int
    deep_clean: int
    accepted_deep_ids: list[int]
    geometry_status: str


@dataclass
class CameraDetectionRun:
    detections_by_index: dict[
        int,
        dict[int, np.ndarray],
    ]

    frame_indices_by_index: dict[
        int,
        int,
    ]

    audits: list[FrameAudit]


def run_camera_sequence(
    video_path: str | Path,
    detector,
    logical_frame_ids,
    frame_offset: int,
    camera_name: str,
) -> CameraDetectionRun:
    """
    Run NanoDeepChArUco over one camera using a shared
    logical stereo schedule.

    For each logical detection index t:

        physical_frame = t + frame_offset

    The logical index is preserved as detection_idx so
    multiple cameras can be aligned correctly by CalibCam.
    """

    video_path = (
        Path(video_path)
        .expanduser()
        .resolve()
    )

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    detections_by_index = {}
    frame_indices_by_index = {}
    audits = []

    try:
        for logical_idx_raw in logical_frame_ids:
            logical_idx = int(
                logical_idx_raw
            )

            physical_idx = (
                physical_frame_index(
                    logical_idx,
                    frame_offset,
                )
            )

            if physical_idx < 0:
                raise RuntimeError(
                    "Scheduler produced a negative "
                    f"physical frame: {physical_idx}"
                )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                physical_idx,
            )

            ok, frame = cap.read()

            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not read {camera_name} "
                    f"physical frame {physical_idx} "
                    f"for logical index {logical_idx}"
                )

            tag = (
                f"{camera_name}_"
                f"logical_{logical_idx:06d}_"
                f"physical_{physical_idx:06d}"
            )

            result = detector.process_frame(
                frame,
                tag=tag,
            )

            corners = {
                int(idx): np.asarray(
                    xy,
                    dtype=np.float32,
                )
                for idx, xy
                in result.corners.items()
            }

            audits.append(
                FrameAudit(
                    detection_idx=logical_idx,
                    physical_frame_idx=physical_idx,
                    status=result.status,
                    corner_count=len(corners),
                    nano_source=result.nano_source,
                    nano_base=result.nano_base,
                    deep_ran=result.deep_ran,
                    deep_raw=result.deep_raw,
                    deep_clean=result.deep_clean,
                    accepted_deep_ids=list(
                        result.accepted_deep_ids
                    ),
                    geometry_status=(
                        result.geometry_status
                    ),
                )
            )

            print(
                f"{camera_name} "
                f"logical={logical_idx} "
                f"physical={physical_idx} "
                f"status={result.status} "
                f"corners={len(corners)}"
            )

            if not corners:
                continue

            detections_by_index[
                logical_idx
            ] = corners

            frame_indices_by_index[
                logical_idx
            ] = physical_idx

    finally:
        cap.release()

    return CameraDetectionRun(
        detections_by_index=(
            detections_by_index
        ),
        frame_indices_by_index=(
            frame_indices_by_index
        ),
        audits=audits,
    )


def save_camera_run_payload(
    run: CameraDetectionRun,
    output_path: str | Path,
    expected_marker_ids: list[int] | None = None,
) -> dict:
    payload = build_calibcam_payload(
        detections_by_index=(
            run.detections_by_index
        ),
        frame_indices_by_index=(
            run.frame_indices_by_index
        ),
        expected_marker_ids=(
            expected_marker_ids
        ),
    )

    save_calibcam_payload(
        output_path,
        payload,
    )

    return payload
