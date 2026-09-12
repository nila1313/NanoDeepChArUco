from __future__ import annotations


def build_auto_sync_detection_maps(
    frames_start: int,
    frames_end: int,
    frames_step: int,
    offsets: list[int],
    left_frame_count: int | None = None,
    right_frame_count: int | None = None,
) -> tuple[
    dict[int, int],
    dict[int, int],
]:
    """
    Build physical-frame detection schedules for automatic
    stereo synchronization.

    Left camera defines the reference timeline.

    Left detection keys:
        physical frame -> same physical frame

    Right detection keys:
        every physical frame required by at least one
        candidate pairing:

            right_frame = left_frame + offset

    The identity mappings are intentional. During the
    synchronization-discovery pass, detections are indexed
    by their physical frame number.
    """

    if frames_step <= 0:
        raise ValueError(
            "frames_step must be greater than zero"
        )

    if frames_end <= frames_start:
        raise ValueError(
            "frames_end must be greater than frames_start"
        )

    if not offsets:
        raise ValueError(
            "At least one synchronization offset is required"
        )

    left_frames = list(
        range(
            int(frames_start),
            int(frames_end),
            int(frames_step),
        )
    )

    if left_frame_count is not None:
        left_frames = [
            frame
            for frame in left_frames
            if 0 <= frame < left_frame_count
        ]
    else:
        left_frames = [
            frame
            for frame in left_frames
            if frame >= 0
        ]

    right_frames = set()

    for left_frame in left_frames:
        for offset in offsets:
            right_frame = (
                int(left_frame)
                + int(offset)
            )

            if right_frame < 0:
                continue

            if (
                right_frame_count is not None
                and right_frame
                >= right_frame_count
            ):
                continue

            right_frames.add(
                right_frame
            )

    left_map = {
        frame: frame
        for frame in left_frames
    }

    right_map = {
        frame: frame
        for frame in sorted(
            right_frames
        )
    }

    return (
        left_map,
        right_map,
    )
