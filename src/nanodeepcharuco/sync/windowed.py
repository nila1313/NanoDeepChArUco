from __future__ import annotations

from dataclasses import dataclass

from .epipolar import (
    EpipolarOffsetScore,
    search_epipolar_offsets,
)


@dataclass(frozen=True)
class WindowSyncResult:
    start_frame: int
    end_frame: int
    best_offset: int | None
    best_score: EpipolarOffsetScore | None
    second_score: EpipolarOffsetScore | None
    inlier_ratio_gap: float
    n_left_frames: int
    n_right_frames: int


def _slice_frames(
    detections_by_frame: dict,
    start_frame: int,
    end_frame: int,
) -> dict:
    return {
        frame_idx: detections
        for frame_idx, detections
        in detections_by_frame.items()
        if start_frame
        <= frame_idx
        < end_frame
    }


def search_windowed_offsets(
    left_by_frame: dict,
    right_by_frame: dict,
    offsets: list[int],
    window_size: int,
    window_step: int | None = None,
    min_shared_per_frame: int = 6,
    ransac_threshold_px: float = 1.5,
    min_frame_pairs: int = 4,
) -> list[WindowSyncResult]:
    if window_size <= 0:
        raise ValueError(
            "window_size must be > 0"
        )

    if window_step is None:
        window_step = window_size

    if window_step <= 0:
        raise ValueError(
            "window_step must be > 0"
        )

    left_frames = sorted(
        left_by_frame.keys()
    )

    if not left_frames:
        return []

    global_start = min(left_frames)
    global_end = max(left_frames) + 1

    min_offset = min(
        int(offset)
        for offset in offsets
    )

    max_offset = max(
        int(offset)
        for offset in offsets
    )

    results = []

    start = global_start

    while start < global_end:
        end = start + window_size

        left_window = _slice_frames(
            left_by_frame,
            start,
            end,
        )

        right_window = _slice_frames(
            right_by_frame,
            start + min_offset,
            end + max_offset,
        )

        scores = search_epipolar_offsets(
            left_by_frame=left_window,
            right_by_frame=right_window,
            offsets=offsets,
            min_shared_per_frame=(
                min_shared_per_frame
            ),
            ransac_threshold_px=(
                ransac_threshold_px
            ),
        )

        valid_scores = [
            score
            for score in scores
            if (
                score.valid
                and score.n_frame_pairs
                >= min_frame_pairs
            )
        ]

        if not valid_scores:
            results.append(
                WindowSyncResult(
                    start_frame=start,
                    end_frame=end,
                    best_offset=None,
                    best_score=None,
                    second_score=None,
                    inlier_ratio_gap=0.0,
                    n_left_frames=len(
                        left_window
                    ),
                    n_right_frames=len(
                        right_window
                    ),
                )
            )

            start += window_step
            continue

        best = valid_scores[0]

        second = (
            valid_scores[1]
            if len(valid_scores) > 1
            else None
        )

        gap = (
            best.inlier_ratio
            - second.inlier_ratio
            if second is not None
            else best.inlier_ratio
        )

        results.append(
            WindowSyncResult(
                start_frame=start,
                end_frame=end,
                best_offset=(
                    best.offset
                ),
                best_score=best,
                second_score=second,
                inlier_ratio_gap=float(
                    gap
                ),
                n_left_frames=len(
                    left_window
                ),
                n_right_frames=len(
                    right_window
                ),
            )
        )

        start += window_step

    return results
