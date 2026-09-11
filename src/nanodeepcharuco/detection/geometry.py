from __future__ import annotations

import cv2
import numpy as np


MIN_UNIQUE_ROWS = 2
MIN_UNIQUE_COLS = 2

MIN_HOMOGRAPHY_ANCHORS = 4
DEFAULT_MAX_GEOM_ERROR_EQ_PX = 5.0

DEEP_SELF_MIN_POINTS = 6
DEEP_SELF_MIN_INLIERS = 5
DEEP_SELF_MIN_ROWS = 2
DEEP_SELF_MIN_COLS = 2
DEFAULT_DEEP_SELF_RANSAC_EQ_PX = 5.0


def inner_grid_width(
    board_width: int,
) -> int:
    width = int(board_width) - 1

    if width <= 0:
        raise ValueError(
            "board_width must be at least 2"
        )

    return width


def board_xy(
    corner_id: int,
    board_width: int,
) -> np.ndarray:
    """
    Convert a ChArUco corner ID into its 2-D board-grid
    coordinate.

    ChArUco has board_width - 1 inner corner columns.
    """

    grid_width = inner_grid_width(
        board_width
    )

    corner_id = int(corner_id)

    row = corner_id // grid_width
    col = corner_id % grid_width

    return np.asarray(
        [
            float(col),
            float(row),
        ],
        dtype=np.float32,
    )


def spread_ok(
    ids,
    board_width: int,
    minimum_rows: int = MIN_UNIQUE_ROWS,
    minimum_cols: int = MIN_UNIQUE_COLS,
) -> bool:
    ids = [
        int(idx)
        for idx in ids
    ]

    if not ids:
        return False

    grid_width = inner_grid_width(
        board_width
    )

    rows = {
        idx // grid_width
        for idx in ids
    }

    cols = {
        idx % grid_width
        for idx in ids
    }

    return (
        len(rows) >= minimum_rows
        and len(cols) >= minimum_cols
    )


def estimate_homography(
    corner_dict: dict[int, np.ndarray],
    board_width: int,
):
    if (
        len(corner_dict)
        < MIN_HOMOGRAPHY_ANCHORS
    ):
        return None

    ids = sorted(
        corner_dict
    )

    object_xy = np.asarray(
        [
            board_xy(
                idx,
                board_width,
            )
            for idx in ids
        ],
        dtype=np.float32,
    )

    image_xy = np.asarray(
        [
            corner_dict[idx]
            for idx in ids
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(
        object_xy,
        image_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    return H


def predicted_from_homography(
    H: np.ndarray,
    corner_id: int,
    board_width: int,
) -> np.ndarray:
    p = board_xy(
        corner_id,
        board_width,
    )

    src = np.asarray(
        [
            [
                [
                    p[0],
                    p[1],
                ]
            ]
        ],
        dtype=np.float32,
    )

    dst = cv2.perspectiveTransform(
        src,
        H,
    )

    return (
        dst.reshape(2)
        .astype(np.float32)
    )


def equivalent_error(
    observed_xy: np.ndarray,
    predicted_xy: np.ndarray,
    frame_shape,
    deep_input_width: int,
    deep_input_height: int,
) -> float:
    h, w = frame_shape[:2]

    dx = (
        float(observed_xy[0])
        - float(predicted_xy[0])
    )

    dy = (
        float(observed_xy[1])
        - float(predicted_xy[1])
    )

    dx_eq = dx * (
        float(deep_input_width)
        / float(w)
    )

    dy_eq = dy * (
        float(deep_input_height)
        / float(h)
    )

    return float(
        np.hypot(
            dx_eq,
            dy_eq,
        )
    )


def fallback_recover(
    nano_dict: dict[int, np.ndarray],
    deep_dict: dict[int, np.ndarray],
    frame_shape,
    deep_input_width: int,
    deep_input_height: int,
    board_width: int,
    max_geom_error_eq_px: float = (
        DEFAULT_MAX_GEOM_ERROR_EQ_PX
    ),
):
    """
    Begin with Nano detections.

    Deep may only add IDs missing from Nano whose
    locations are geometrically consistent with the
    board homography estimated from Nano anchors.
    """

    result = {
        int(idx): np.asarray(
            xy,
            dtype=np.float32,
        )
        for idx, xy
        in nano_dict.items()
    }

    H = estimate_homography(
        nano_dict,
        board_width,
    )

    if H is None:
        return (
            result,
            [],
            {},
            "NO_HOMOGRAPHY",
        )

    accepted = []
    rejected = {}

    for idx, deep_xy in sorted(
        deep_dict.items()
    ):
        idx = int(idx)

        if idx in result:
            continue

        expected_xy = (
            predicted_from_homography(
                H,
                idx,
                board_width,
            )
        )

        err = equivalent_error(
            deep_xy,
            expected_xy,
            frame_shape,
            deep_input_width,
            deep_input_height,
        )

        if (
            err
            <= float(
                max_geom_error_eq_px
            )
        ):
            result[idx] = np.asarray(
                deep_xy,
                dtype=np.float32,
            )

            accepted.append(
                idx
            )

        else:
            rejected[idx] = err

    return (
        result,
        accepted,
        rejected,
        "HOMOGRAPHY_OK",
    )


def deep_self_spread_ok(
    ids,
    board_width: int,
) -> bool:
    return spread_ok(
        ids,
        board_width,
        minimum_rows=DEEP_SELF_MIN_ROWS,
        minimum_cols=DEEP_SELF_MIN_COLS,
    )


def native_ransac_threshold(
    frame_shape,
    deep_input_width: int,
    deep_input_height: int,
    ransac_eq_px: float = (
        DEFAULT_DEEP_SELF_RANSAC_EQ_PX
    ),
) -> float:
    h, w = frame_shape[:2]

    sx = (
        float(w)
        / float(deep_input_width)
    )

    sy = (
        float(h)
        / float(deep_input_height)
    )

    scale = 0.5 * (
        sx + sy
    )

    return (
        float(ransac_eq_px)
        * scale
    )


def deep_self_verify(
    deep_dict: dict[int, np.ndarray],
    frame_shape,
    deep_input_width: int,
    deep_input_height: int,
    board_width: int,
    ransac_eq_px: float = (
        DEFAULT_DEEP_SELF_RANSAC_EQ_PX
    ),
):
    if (
        len(deep_dict)
        < DEEP_SELF_MIN_POINTS
    ):
        return (
            {},
            [],
            "SELF_TOO_FEW_DEEP",
        )

    ids = sorted(
        int(idx)
        for idx in deep_dict
    )

    if not deep_self_spread_ok(
        ids,
        board_width,
    ):
        return (
            {},
            [],
            "SELF_BAD_SPREAD",
        )

    object_xy = np.asarray(
        [
            board_xy(
                idx,
                board_width,
            )
            for idx in ids
        ],
        dtype=np.float32,
    )

    image_xy = np.asarray(
        [
            deep_dict[idx]
            for idx in ids
        ],
        dtype=np.float32,
    )

    ransac_px = (
        native_ransac_threshold(
            frame_shape,
            deep_input_width,
            deep_input_height,
            ransac_eq_px=ransac_eq_px,
        )
    )

    H, mask = cv2.findHomography(
        object_xy,
        image_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_px,
    )

    if (
        H is None
        or mask is None
    ):
        return (
            {},
            [],
            "SELF_H_FAIL",
        )

    mask = (
        mask.reshape(-1)
        .astype(bool)
    )

    inlier_ids = [
        idx
        for idx, keep
        in zip(ids, mask)
        if keep
    ]

    if (
        len(inlier_ids)
        < DEEP_SELF_MIN_INLIERS
    ):
        return (
            {},
            inlier_ids,
            "SELF_TOO_FEW_INLIERS",
        )

    if not deep_self_spread_ok(
        inlier_ids,
        board_width,
    ):
        return (
            {},
            inlier_ids,
            "SELF_INLIER_BAD_SPREAD",
        )

    verified = {
        idx: np.asarray(
            deep_dict[idx],
            dtype=np.float32,
        )
        for idx in inlier_ids
    }

    return (
        verified,
        inlier_ids,
        "SELF_OK",
    )
