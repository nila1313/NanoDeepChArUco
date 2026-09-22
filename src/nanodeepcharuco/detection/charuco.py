from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import sys

# Compatibility for board.npy files created with newer NumPy versions.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

from .aruco_nano import MarkerDetection


@dataclass(frozen=True)
class CharucoDetection:
    corner_id: int
    point: np.ndarray


def load_board_parameters(board_path: str | Path) -> dict:
    board_path = Path(board_path)

    if not board_path.is_file():
        raise FileNotFoundError(f"Board file not found: {board_path}")

    params = np.load(board_path, allow_pickle=True).item()

    required = {
        "boardWidth",
        "boardHeight",
        "square_size_real",
        "marker_size_real",
        "dictionary_type",
    }

    missing = required - set(params.keys())

    if missing:
        raise ValueError(
            f"Board configuration missing keys: {sorted(missing)}"
        )

    return params


def build_charuco_board(board_path: str | Path):
    params = load_board_parameters(board_path)

    dictionary = cv2.aruco.getPredefinedDictionary(
        int(params["dictionary_type"])
    )

    board = cv2.aruco.CharucoBoard(
        (
            int(params["boardWidth"]),
            int(params["boardHeight"]),
        ),
        float(params["square_size_real"]),
        float(params["marker_size_real"]),
        dictionary,
    )

    return board


def interpolate_charuco_from_nano(
    image: np.ndarray,
    marker_detections: list[MarkerDetection],
    board,
) -> list[CharucoDetection]:
    """
    Convert ArUco Nano marker detections into standard ChArUco corners.

    ArUco Nano performs marker detection.
    OpenCV is used only for the ChArUco board geometry/interpolation.
    """

    if image is None:
        raise ValueError("Input image is None")

    if not marker_detections:
        return []

    marker_corners = []
    marker_ids = []

    for detection in marker_detections:
        corners = np.asarray(
            detection.corners,
            dtype=np.float32,
        ).reshape(1, 4, 2)

        marker_corners.append(corners)
        marker_ids.append(detection.marker_id)

    marker_ids = np.asarray(
        marker_ids,
        dtype=np.int32,
    ).reshape(-1, 1)

    retval, charuco_corners, charuco_ids = (
        cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            image,
            board,
        )
    )

    if (
        retval is None
        or retval <= 0
        or charuco_corners is None
        or charuco_ids is None
    ):
        return []

    points = charuco_corners.reshape(-1, 2)
    ids = charuco_ids.reshape(-1)

    detections = [
        CharucoDetection(
            corner_id=int(corner_id),
            point=np.asarray(point, dtype=np.float32),
        )
        for corner_id, point in zip(ids, points)
    ]

    detections.sort(key=lambda detection: detection.corner_id)

    return detections
