from __future__ import annotations

from dataclasses import dataclass

from .windowed import WindowSyncResult


@dataclass(frozen=True)
class SyncSegment:
    start_frame: int
    end_frame: int
    offset: int
    n_support_windows: int
    mean_inlier_ratio: float
    mean_ratio_gap: float


def select_stable_windows(
    results: list[WindowSyncResult],
    min_inlier_ratio: float = 0.20,
    min_ratio_gap: float = 0.05,
) -> list[WindowSyncResult]:
    accepted = []

    for result in results:
        score = result.best_score

        if score is None:
            continue

        if score.inlier_ratio < min_inlier_ratio:
            continue

        if result.inlier_ratio_gap < min_ratio_gap:
            continue

        accepted.append(result)

    return accepted


def build_sync_segments(
    results: list[WindowSyncResult],
    min_inlier_ratio: float = 0.20,
    min_ratio_gap: float = 0.05,
    min_persistence: int = 2,
    max_gap_frames: int = 400,
) -> list[SyncSegment]:
    stable = select_stable_windows(
        results,
        min_inlier_ratio=min_inlier_ratio,
        min_ratio_gap=min_ratio_gap,
    )

    if not stable:
        return []

    groups: list[list[WindowSyncResult]] = []

    current_group = [stable[0]]

    for result in stable[1:]:
        previous = current_group[-1]

        same_offset = (
            result.best_offset
            == previous.best_offset
        )

        gap_frames = max(
            0,
            result.start_frame
            - previous.end_frame,
        )

        close_enough = (
            gap_frames
            <= max_gap_frames
        )

        if same_offset and close_enough:
            current_group.append(result)
        else:
            groups.append(current_group)
            current_group = [result]

    groups.append(current_group)

    segments = []

    for group in groups:
        if len(group) < min_persistence:
            continue

        ratios = [
            item.best_score.inlier_ratio
            for item in group
            if item.best_score is not None
        ]

        gaps = [
            item.inlier_ratio_gap
            for item in group
        ]

        segments.append(
            SyncSegment(
                start_frame=group[0].start_frame,
                end_frame=group[-1].end_frame,
                offset=int(
                    group[0].best_offset
                ),
                n_support_windows=len(group),
                mean_inlier_ratio=(
                    sum(ratios) / len(ratios)
                ),
                mean_ratio_gap=(
                    sum(gaps) / len(gaps)
                ),
            )
        )

    return segments
