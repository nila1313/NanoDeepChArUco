from __future__ import annotations

import cv2
import numpy as np


MIN_UNIQUE_ROWS = 2
MIN_UNIQUE_COLS = 2

MIN_HOMOGRAPHY_ANCHORS = 4
MAX_GEOM_ERROR_EQ_PX = 1.5

DEEP_SELF_MIN_POINTS = 6
DEEP_SELF_MIN_INLIERS = 5
DEEP_SELF_MIN_ROWS = 2
DEEP_SELF_MIN_COLS = 2
DEEP_SELF_RANSAC_EQ_PX = 5.0


def board_xy(corner_id: int, grid_width: int) -> np.ndarray:
    row = int(corner_id) // grid_width
    col = int(corner_id) % grid_width

    return np.asarray(
        [float(col), float(row)],
        dtype=np.float32,
    )


def spread_ok(ids, grid_width: int) -> bool:
    ids = list(ids)

    if not ids:
        return False

    rows = {
        int(idx) // grid_width
        for idx in ids
    }

    cols = {
        int(idx) % grid_width
        for idx in ids
    }

    return (
        len(rows) >= MIN_UNIQUE_ROWS
        and len(cols) >= MIN_UNIQUE_COLS
    )


def estimate_homography(
    corner_dict: dict[int, np.ndarray],
    grid_width: int,
):
    if len(corner_dict) < MIN_HOMOGRAPHY_ANCHORS:
        return None

    ids = sorted(corner_dict)

    object_xy = np.asarray(
        [
            board_xy(idx, grid_width)
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
    grid_width: int,
) -> np.ndarray:
    p = board_xy(corner_id, grid_width)

    src = np.asarray(
        [[[p[0], p[1]]]],
        dtype=np.float32,
    )

    dst = cv2.perspectiveTransform(
        src,
        H,
    )

    return dst.reshape(2).astype(np.float32)


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
        float(deep_input_width) / float(w)
    )

    dy_eq = dy * (
        float(deep_input_height) / float(h)
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
    grid_width: int,
):
    """
    Start with Nano.

    Deep may ONLY add IDs that Nano did not detect and whose
    position is geometrically consistent with the board geometry
    estimated from Nano anchors.
    """

    result = {
        idx: np.asarray(
            xy,
            dtype=np.float32,
        )
        for idx, xy in nano_dict.items()
    }

    h, w = frame_shape[:2]

    H = estimate_homography(
        nano_dict,
        grid_width,
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
        if idx in result:
            continue

        expected_xy = predicted_from_homography(
            H,
            idx,
            grid_width,
        )

        err = equivalent_error(
            deep_xy,
            expected_xy,
            frame_shape,
            deep_input_width,
            deep_input_height,
        )

        if err <= MAX_GEOM_ERROR_EQ_PX:
            result[idx] = deep_xy
            accepted.append(idx)
        else:
            rejected[idx] = err

    return (
        result,
        accepted,
        rejected,
        "HOMOGRAPHY_OK",
    )


def deep_self_spread_ok(ids, grid_width: int) -> bool:
    ids = list(ids)

    rows = {
        int(idx) // grid_width
        for idx in ids
    }

    cols = {
        int(idx) % grid_width
        for idx in ids
    }

    return (
        len(rows) >= DEEP_SELF_MIN_ROWS
        and len(cols) >= DEEP_SELF_MIN_COLS
    )


def native_ransac_threshold(
    frame_shape,
    deep_input_width: int,
    deep_input_height: int,
) -> float:
    h, w = frame_shape[:2]

    sx = float(w) / float(deep_input_width)
    sy = float(h) / float(deep_input_height)

    scale = 0.5 * (sx + sy)

    return (
        DEEP_SELF_RANSAC_EQ_PX
        * scale
    )


def deep_self_verify(
    deep_dict: dict[int, np.ndarray],
    frame_shape,
    deep_input_width: int,
    deep_input_height: int,
    grid_width: int,
):
    if len(deep_dict) < DEEP_SELF_MIN_POINTS:
        return (
            {},
            [],
            "SELF_TOO_FEW_DEEP",
        )

    ids = sorted(deep_dict)

    if not deep_self_spread_ok(ids, grid_width):
        return (
            {},
            [],
            "SELF_BAD_SPREAD",
        )

    object_xy = np.asarray(
        [
            board_xy(idx, grid_width)
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

    ransac_px = native_ransac_threshold(
        frame_shape,
        deep_input_width,
        deep_input_height,
    )

    H, mask = cv2.findHomography(
        object_xy,
        image_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_px,
    )

    if H is None or mask is None:
        return (
            {},
            [],
            "SELF_H_FAIL",
        )

    mask = mask.reshape(-1).astype(bool)

    inlier_ids = [
        idx
        for idx, keep in zip(ids, mask)
        if keep
    ]

    if len(inlier_ids) < DEEP_SELF_MIN_INLIERS:
        return (
            {},
            inlier_ids,
            "SELF_TOO_FEW_INLIERS",
        )

    if not deep_self_spread_ok(
        inlier_ids,
        grid_width,
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
