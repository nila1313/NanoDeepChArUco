from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EpipolarOffsetScore:
    offset: int
    n_frame_pairs: int
    n_correspondences: int
    n_inliers: int
    inlier_ratio: float
    median_sampson_error: float
    mean_sampson_error: float

    @property
    def valid(self) -> bool:
        return (
            self.n_correspondences >= 8
            and self.n_inliers >= 8
            and np.isfinite(
                self.median_sampson_error
            )
        )


def _sampson_errors(
    F: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
) -> np.ndarray:
    x1 = np.column_stack(
        [
            pts_left,
            np.ones(
                len(pts_left),
                dtype=np.float64,
            ),
        ]
    )

    x2 = np.column_stack(
        [
            pts_right,
            np.ones(
                len(pts_right),
                dtype=np.float64,
            ),
        ]
    )

    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T

    numerator = np.sum(
        x2 * Fx1,
        axis=1,
    ) ** 2

    denominator = (
        Fx1[:, 0] ** 2
        + Fx1[:, 1] ** 2
        + Ftx2[:, 0] ** 2
        + Ftx2[:, 1] ** 2
    )

    valid = denominator > 1e-12

    errors = np.full(
        len(pts_left),
        np.inf,
        dtype=np.float64,
    )

    errors[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return errors


def score_epipolar_offset(
    left_by_frame: dict[
        int,
        dict[int, np.ndarray],
    ],
    right_by_frame: dict[
        int,
        dict[int, np.ndarray],
    ],
    offset: int,
    min_shared_per_frame: int = 6,
    ransac_threshold_px: float = 1.5,
) -> EpipolarOffsetScore:
    left_points = []
    right_points = []

    n_frame_pairs = 0

    for left_frame in sorted(
        left_by_frame.keys()
    ):
        right_frame = (
            left_frame + offset
        )

        if (
            right_frame
            not in right_by_frame
        ):
            continue

        left = left_by_frame[
            left_frame
        ]

        right = right_by_frame[
            right_frame
        ]

        shared_ids = sorted(
            set(left.keys())
            & set(right.keys())
        )

        if (
            len(shared_ids)
            < min_shared_per_frame
        ):
            continue

        n_frame_pairs += 1

        for marker_id in shared_ids:
            left_points.append(
                left[marker_id]
            )

            right_points.append(
                right[marker_id]
            )

    n_correspondences = len(
        left_points
    )

    if n_correspondences < 8:
        return EpipolarOffsetScore(
            offset=offset,
            n_frame_pairs=n_frame_pairs,
            n_correspondences=(
                n_correspondences
            ),
            n_inliers=0,
            inlier_ratio=0.0,
            median_sampson_error=(
                float("inf")
            ),
            mean_sampson_error=(
                float("inf")
            ),
        )

    pts_left = np.asarray(
        left_points,
        dtype=np.float64,
    ).reshape(-1, 2)

    pts_right = np.asarray(
        right_points,
        dtype=np.float64,
    ).reshape(-1, 2)

    F, mask = cv2.findFundamentalMat(
        pts_left,
        pts_right,
        cv2.FM_RANSAC,
        ransac_threshold_px,
        0.999,
    )

    if (
        F is None
        or mask is None
        or np.asarray(F).shape
        != (3, 3)
    ):
        return EpipolarOffsetScore(
            offset=offset,
            n_frame_pairs=n_frame_pairs,
            n_correspondences=(
                n_correspondences
            ),
            n_inliers=0,
            inlier_ratio=0.0,
            median_sampson_error=(
                float("inf")
            ),
            mean_sampson_error=(
                float("inf")
            ),
        )

    F = np.asarray(
        F,
        dtype=np.float64,
    )

    inliers = (
        mask.reshape(-1)
        .astype(bool)
    )

    errors = _sampson_errors(
        F,
        pts_left,
        pts_right,
    )

    errors_in = errors[inliers]

    errors_in = errors_in[
        np.isfinite(errors_in)
    ]

    if len(errors_in) == 0:
        median_error = float("inf")
        mean_error = float("inf")
    else:
        median_error = float(
            np.median(errors_in)
        )

        mean_error = float(
            np.mean(errors_in)
        )

    n_inliers = int(
        np.sum(inliers)
    )

    return EpipolarOffsetScore(
        offset=offset,
        n_frame_pairs=n_frame_pairs,
        n_correspondences=(
            n_correspondences
        ),
        n_inliers=n_inliers,
        inlier_ratio=(
            n_inliers
            / n_correspondences
        ),
        median_sampson_error=(
            median_error
        ),
        mean_sampson_error=(
            mean_error
        ),
    )


def search_epipolar_offsets(
    left_by_frame: dict[
        int,
        dict[int, np.ndarray],
    ],
    right_by_frame: dict[
        int,
        dict[int, np.ndarray],
    ],
    offsets: list[int],
    min_shared_per_frame: int = 6,
    ransac_threshold_px: float = 1.5,
) -> list[EpipolarOffsetScore]:
    results = [
        score_epipolar_offset(
            left_by_frame=(
                left_by_frame
            ),
            right_by_frame=(
                right_by_frame
            ),
            offset=offset,
            min_shared_per_frame=(
                min_shared_per_frame
            ),
            ransac_threshold_px=(
                ransac_threshold_px
            ),
        )
        for offset in offsets
    ]

    return sorted(
        results,
        key=lambda result: (
            -result.inlier_ratio,
            result.median_sampson_error,
            -result.n_inliers,
        ),
    )
