from __future__ import annotations

from dataclasses import dataclass

from .offset_search import build_motion_signal


@dataclass(frozen=True)
class StableBoardSegment:
    start_frame: int
    end_frame: int
    n_frames: int
    max_motion_px: float
    mean_motion_px: float


def find_stable_board_segments(
    detections_by_frame: dict,
    motion_threshold_px: float = 2.0,
    min_stable_frames: int = 3,
    min_shared_corners: int = 4,
) -> list[StableBoardSegment]:
    """
    Find intervals where the detected ChArUco board moves only slightly.

    Motion is measured as the median displacement of shared ChArUco
    corners between consecutive physical video frames.

    A stable interval requires at least min_stable_frames consecutive
    physical frames whose board motion stays below motion_threshold_px.
    """

    if motion_threshold_px < 0:
        raise ValueError(
            "motion_threshold_px cannot be negative"
        )

    if min_stable_frames < 2:
        raise ValueError(
            "min_stable_frames must be at least 2"
        )

    signal = build_motion_signal(
        detections_by_frame,
        min_shared=min_shared_corners,
    )

    if not signal:
        return []

    # signal[t] represents board motion from frame t-1 -> t.
    stable_transitions = sorted(
        frame
        for frame, motion in signal.items()
        if motion <= motion_threshold_px
    )

    if not stable_transitions:
        return []

    groups: list[list[int]] = []
    current = [stable_transitions[0]]

    for frame in stable_transitions[1:]:
        if frame == current[-1] + 1:
            current.append(frame)
        else:
            groups.append(current)
            current = [frame]

    groups.append(current)

    segments = []

    for group in groups:
        # N stable transitions describe N+1 stable physical frames.
        n_frames = len(group) + 1

        if n_frames < min_stable_frames:
            continue

        transition_motion = [
            float(signal[frame])
            for frame in group
        ]

        segments.append(
            StableBoardSegment(
                start_frame=group[0] - 1,
                end_frame=group[-1] + 1,
                n_frames=n_frames,
                max_motion_px=max(
                    transition_motion
                ),
                mean_motion_px=(
                    sum(transition_motion)
                    / len(transition_motion)
                ),
            )
        )

    return segments


def frame_is_in_stable_segment(
    frame_idx: int,
    segments: list[StableBoardSegment],
) -> bool:
    return any(
        segment.start_frame
        <= int(frame_idx)
        < segment.end_frame
        for segment in segments
    )


def filter_pairs_by_board_stability(
    pairs,
    left_segments: list[StableBoardSegment],
    right_segments: list[StableBoardSegment],
):
    """
    Keep a synchronized stereo pair only when the board is inside
    a stable interval in both physical videos.
    """

    return [
        pair
        for pair in pairs
        if (
            frame_is_in_stable_segment(
                pair.left_frame,
                left_segments,
            )
            and frame_is_in_stable_segment(
                pair.right_frame,
                right_segments,
            )
        )
    ]
