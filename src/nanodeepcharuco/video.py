from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    frame_count: int
    width: int
    height: int
    fps: float


def inspect_video(path: str) -> VideoInfo:
    video_path = Path(path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    try:
        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )
    finally:
        cap.release()

    if frame_count <= 0:
        raise RuntimeError(
            f"Invalid frame count for: {video_path}"
        )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video dimensions for: {video_path}"
        )

    return VideoInfo(
        path=str(video_path.resolve()),
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
    )


def inspect_videos(
    paths: list[str],
) -> list[VideoInfo]:
    return [
        inspect_video(path)
        for path in paths
    ]


def build_logical_frame_ids(
    video_infos: list[VideoInfo],
    frames_start: int,
    frames_end: int | None,
    frames_step: int,
    frames_offsets: list[int],
) -> np.ndarray:
    """
    Build the common logical frame schedule.

    NanoDeepChArUco uses explicit offset semantics:

        physical_frame[camera] = logical_frame + offset[camera]

    Examples:

        offsets [0, 0]
            cam0(t)   <-> cam1(t)

        offsets [0, 1]
            cam0(t)   <-> cam1(t + 1)

        offsets [1, 0]
            cam0(t + 1) <-> cam1(t)

    The logical range is automatically restricted so that
    every physical frame index is valid for every camera.

    frames_end follows Python/CalibCam-style exclusive-end
    semantics.
    """

    if frames_step <= 0:
        raise ValueError(
            "frames_step must be greater than zero."
        )

    if len(video_infos) != len(frames_offsets):
        raise ValueError(
            "Number of videos and offsets must match."
        )

    if frames_start < 0:
        raise ValueError(
            "frames_start must be >= 0."
        )

    if not video_infos:
        raise ValueError(
            "At least one video is required."
        )

    # Lower bound:
    #
    #     logical + offset >= 0
    #
    # therefore:
    #
    #     logical >= -offset
    #
    common_start = max(
        [frames_start]
        + [
            -int(offset)
            for offset in frames_offsets
        ]
    )

    # Exclusive upper bound:
    #
    #     logical + offset < frame_count
    #
    # therefore:
    #
    #     logical < frame_count - offset
    #
    common_stop = min(
        int(info.frame_count) - int(offset)
        for info, offset in zip(
            video_infos,
            frames_offsets,
        )
    )

    if frames_end is not None:
        if frames_end < 0:
            raise ValueError(
                "frames_end must be >= 0."
            )

        common_stop = min(
            common_stop,
            int(frames_end),
        )

    if common_stop <= common_start:
        raise ValueError(
            "No valid common frames remain after "
            "applying frame range and offsets."
        )

    return np.arange(
        common_start,
        common_stop,
        frames_step,
        dtype=np.int64,
    )

def physical_frame_index(
    logical_frame: int,
    offset: int,
) -> int:
    return int(
        logical_frame + offset
    )


def read_frame(
    video_path: str,
    frame_index: int,
) -> np.ndarray:
    if frame_index < 0:
        raise ValueError(
            f"Frame index must be >= 0, got {frame_index}"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    try:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            raise RuntimeError(
                f"Could not read frame {frame_index} "
                f"from {video_path}"
            )

        return frame

    finally:
        cap.release()
