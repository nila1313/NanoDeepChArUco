from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyncPadding:
    min_offset: int
    max_offset: int
    left_start: int
    left_end: int
    right_start: int
    right_end: int


def compute_sync_padding(
    frames_start: int,
    frames_end: int,
    offsets: list[int],
) -> SyncPadding:
    """
    Compute physical frame ranges needed for automatic
    stereo synchronization.

    frames_end is exclusive.

    The left camera is the reference timeline:

        Left(t)

    Candidate right-camera frames are:

        Right(t + offset)

    Therefore the right physical range must include the
    minimum and maximum candidate offsets.
    """

    if frames_end <= frames_start:
        raise ValueError(
            "frames_end must be greater than frames_start"
        )

    if not offsets:
        raise ValueError(
            "At least one synchronization offset is required"
        )

    min_offset = min(
        int(offset)
        for offset in offsets
    )

    max_offset = max(
        int(offset)
        for offset in offsets
    )

    left_start = int(frames_start)
    left_end = int(frames_end)

    right_start = (
        int(frames_start)
        + min_offset
    )

    right_end = (
        int(frames_end)
        + max_offset
    )

    right_start = max(
        0,
        right_start,
    )

    return SyncPadding(
        min_offset=min_offset,
        max_offset=max_offset,
        left_start=left_start,
        left_end=left_end,
        right_start=right_start,
        right_end=right_end,
    )
