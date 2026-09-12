from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OffsetScore:
    offset: int
    n_samples: int
    correlation: float
    normalized_mae: float

    @property
    def valid(self) -> bool:
        return self.n_samples >= 3


def _motion_between_frames(
    first: dict[int, np.ndarray],
    second: dict[int, np.ndarray],
    min_shared: int = 4,
) -> float | None:
    """
    Estimate board image motion between two consecutive
    frames from common ChArUco corners.

    We use the median per-corner displacement magnitude,
    which is robust to individual corner noise.
    """

    shared_ids = sorted(
        set(first.keys()) & set(second.keys())
    )

    if len(shared_ids) < min_shared:
        return None

    p0 = np.asarray(
        [first[idx] for idx in shared_ids],
        dtype=np.float64,
    )

    p1 = np.asarray(
        [second[idx] for idx in shared_ids],
        dtype=np.float64,
    )

    displacement = np.linalg.norm(
        p1 - p0,
        axis=1,
    )

    return float(
        np.median(displacement)
    )


def build_motion_signal(
    detections_by_frame: dict[
        int,
        dict[int, np.ndarray],
    ],
    min_shared: int = 4,
) -> dict[int, float]:
    """
    Build a motion signal.

    A value stored under frame t describes motion from
    frame t-1 to frame t.
    """

    frames = sorted(
        detections_by_frame.keys()
    )

    signal: dict[int, float] = {}

    for previous, current in zip(
        frames[:-1],
        frames[1:],
    ):
        # Synchronization search expects consecutive
        # physical frames.
        if current != previous + 1:
            continue

        motion = _motion_between_frames(
            detections_by_frame[previous],
            detections_by_frame[current],
            min_shared=min_shared,
        )

        if motion is not None:
            signal[current] = motion

    return signal


def score_offset(
    left_signal: dict[int, float],
    right_signal: dict[int, float],
    offset: int,
) -> OffsetScore:
    """
    Positive offset means:

        Left(t) <-> Right(t + offset)
    """

    left_values: list[float] = []
    right_values: list[float] = []

    for left_frame in sorted(
        left_signal.keys()
    ):
        right_frame = left_frame + offset

        if right_frame not in right_signal:
            continue

        left_values.append(
            left_signal[left_frame]
        )

        right_values.append(
            right_signal[right_frame]
        )

    n_samples = len(left_values)

    if n_samples < 3:
        return OffsetScore(
            offset=offset,
            n_samples=n_samples,
            correlation=float("-inf"),
            normalized_mae=float("inf"),
        )

    left = np.asarray(
        left_values,
        dtype=np.float64,
    )

    right = np.asarray(
        right_values,
        dtype=np.float64,
    )

    left_std = float(
        np.std(left)
    )

    right_std = float(
        np.std(right)
    )

    if left_std < 1e-12 or right_std < 1e-12:
        correlation = float("-inf")
        normalized_mae = float("inf")
    else:
        left_z = (
            left - np.mean(left)
        ) / left_std

        right_z = (
            right - np.mean(right)
        ) / right_std

        correlation = float(
            np.corrcoef(
                left_z,
                right_z,
            )[0, 1]
        )

        normalized_mae = float(
            np.mean(
                np.abs(
                    left_z - right_z
                )
            )
        )

    return OffsetScore(
        offset=offset,
        n_samples=n_samples,
        correlation=correlation,
        normalized_mae=normalized_mae,
    )


def search_offsets(
    left_by_frame: dict[
        int,
        dict[int, np.ndarray],
    ],
    right_by_frame: dict[
        int,
        dict[int, np.ndarray],
    ],
    offsets: list[int],
    min_shared: int = 4,
) -> list[OffsetScore]:
    left_signal = build_motion_signal(
        left_by_frame,
        min_shared=min_shared,
    )

    right_signal = build_motion_signal(
        right_by_frame,
        min_shared=min_shared,
    )

    results = [
        score_offset(
            left_signal,
            right_signal,
            offset,
        )
        for offset in offsets
    ]

    return sorted(
        results,
        key=lambda item: (
            -item.correlation,
            item.normalized_mae,
            -item.n_samples,
        ),
    )
