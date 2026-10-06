from __future__ import annotations

from dataclasses import dataclass

from .segments import SyncSegment
from .windowed import WindowSyncResult


@dataclass(frozen=True)
class SynchronizedFramePair:
    left_frame: int
    right_frame: int
    offset: int
    segment_start: int
    segment_end: int


def find_segment_for_frame(
    frame_idx: int,
    segments: list[SyncSegment],
) -> SyncSegment | None:
    for segment in segments:
        if (
            segment.start_frame
            <= frame_idx
            < segment.end_frame
        ):
            return segment

    return None


def find_trusted_window_for_frame(
    frame_idx: int,
    offset: int,
    window_results: list[WindowSyncResult],
    min_inlier_ratio: float,
    min_ratio_gap: float,
) -> WindowSyncResult | None:
    """
    Return a trusted synchronization window covering frame_idx.

    A window is trusted only when:
      - it contains a valid best score,
      - its selected offset matches the stable segment offset,
      - its inlier ratio passes the configured threshold,
      - its best-vs-second-best ratio gap passes the threshold.
    """

    for result in window_results:
        if not (
            result.start_frame
            <= frame_idx
            < result.end_frame
        ):
            continue

        if result.best_score is None:
            continue

        if result.best_offset is None:
            continue

        if int(result.best_offset) != int(offset):
            continue

        if (
            float(result.best_score.inlier_ratio)
            < float(min_inlier_ratio)
        ):
            continue

        if (
            float(result.inlier_ratio_gap)
            < float(min_ratio_gap)
        ):
            continue

        return result

    return None


def build_synchronized_pairs(
    left_frames: list[int],
    right_frames: list[int],
    segments: list[SyncSegment],
    require_right_available: bool = True,
    window_results: list[WindowSyncResult] | None = None,
    min_inlier_ratio: float = 0.0,
    min_ratio_gap: float = 0.0,
) -> list[SynchronizedFramePair]:
    right_frame_set = set(
        int(frame)
        for frame in right_frames
    )

    pairs = []

    for left_frame in sorted(
        int(frame)
        for frame in left_frames
    ):
        segment = find_segment_for_frame(
            left_frame,
            segments,
        )

        if segment is None:
            continue

        # Conservative auto-sync rule:
        # being inside a broad stable segment is not enough.
        # The calibration frame must also lie inside a
        # directly trusted support window for that same offset.
        if window_results is not None:
            trusted_window = (
                find_trusted_window_for_frame(
                    frame_idx=left_frame,
                    offset=segment.offset,
                    window_results=window_results,
                    min_inlier_ratio=(
                        min_inlier_ratio
                    ),
                    min_ratio_gap=(
                        min_ratio_gap
                    ),
                )
            )

            if trusted_window is None:
                continue

        right_frame = (
            left_frame
            + segment.offset
        )

        if (
            require_right_available
            and right_frame
            not in right_frame_set
        ):
            continue

        pairs.append(
            SynchronizedFramePair(
                left_frame=left_frame,
                right_frame=right_frame,
                offset=segment.offset,
                segment_start=(
                    segment.start_frame
                ),
                segment_end=(
                    segment.end_frame
                ),
            )
        )

    return pairs
