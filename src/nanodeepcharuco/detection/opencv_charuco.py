from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .charuco import build_charuco_board, load_board_parameters


MIN_CORNERS = 5
INTER_FRAME_DIST = 3.0


@dataclass(frozen=True)
class OpenCVCharucoDetection:
    corner_id: int
    point: np.ndarray


def check_detections_nondegenerate(
    board_width: int,
    charuco_ids,
    minimum_points: int = MIN_CORNERS,
) -> bool:
    charuco_ids = np.asarray(
        charuco_ids
    ).ravel()

    if len(charuco_ids) < minimum_points:
        return False

    # All corners in one row.
    if charuco_ids[-1] < (
        np.floor(
            charuco_ids[0]
            / (board_width - 1)
        ) + 1
    ) * (board_width - 1):
        return False

    # All corners in one column.
    if np.all(
        np.mod(
            np.diff(charuco_ids),
            board_width - 1,
        ) == 0
    ):
        return False

    return True


class OpenCVCharucoDetector:
    def __init__(
        self,
        board_path: str | Path,
    ):
        self.board_path = (
            Path(board_path)
            .expanduser()
            .resolve()
        )

        self.params = load_board_parameters(
            self.board_path
        )

        self.dictionary = (
            cv2.aruco.getPredefinedDictionary(
                int(
                    self.params[
                        "dictionary_type"
                    ]
                )
            )
        )

        self.board = build_charuco_board(
            self.board_path
        )

        # Match CalibCam initial detection:
        # plain DetectorParameters().
        self.detector_params = (
            cv2.aruco.DetectorParameters()
        )

        self.aruco_detector = (
            cv2.aruco.ArucoDetector(
                self.dictionary,
                self.detector_params,
            )
        )

        # Match CalibCam refineDetectedMarkers options.
        self.refine_params = (
            cv2.aruco.DetectorParameters()
        )

        self.refine_params.adaptiveThreshWinSizeMin = 3
        self.refine_params.adaptiveThreshWinSizeMax = 23
        self.refine_params.adaptiveThreshWinSizeStep = 10

        self.refine_params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )

        self.refine_params.cornerRefinementWinSize = 5
        self.refine_params.cornerRefinementMaxIterations = 30
        self.refine_params.cornerRefinementMinAccuracy = 0.01

        self.refine_params.errorCorrectionRate = 0.3
        self.refine_params.perspectiveRemovePixelPerCell = 8

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[OpenCVCharucoDetection]:

        if frame is None:
            raise ValueError("Input frame is None")

        corners, ids, rejected = (
            self.aruco_detector.detectMarkers(
                frame
            )
        )

        if (
            ids is None
            or len(corners) == 0
        ):
            return []

        corners_ref, ids_ref, _, _ = (
            cv2.aruco.refineDetectedMarkers(
                frame,
                self.board,
                corners,
                ids,
                rejected,
                minRepDistance=3.0,
                errorCorrectionRate=1.0,
                checkAllOrders=True,
                parameters=self.refine_params,
            )
        )

        if (
            ids_ref is None
            or len(corners_ref) == 0
        ):
            return []

        (
            retval,
            charuco_corners,
            charuco_ids,
        ) = cv2.aruco.interpolateCornersCharuco(
            corners_ref,
            ids_ref,
            frame,
            self.board,
            minMarkers=1,
        )

        if (
            retval is None
            or retval <= 0
            or charuco_corners is None
            or charuco_ids is None
        ):
            return []

        ids_flat = charuco_ids.reshape(-1)

        if not check_detections_nondegenerate(
            int(self.params["boardWidth"]),
            ids_flat,
            minimum_points=MIN_CORNERS,
        ):
            return []

        points = (
            charuco_corners
            .reshape(-1, 2)
        )

        detections = [
            OpenCVCharucoDetection(
                corner_id=int(corner_id),
                point=np.asarray(
                    point,
                    dtype=np.float32,
                ),
            )
            for corner_id, point
            in zip(ids_flat, points)
        ]

        detections.sort(
            key=lambda detection:
            detection.corner_id
        )

        return detections

    def detect_dict(
        self,
        frame: np.ndarray,
    ) -> dict[int, np.ndarray]:

        return {
            detection.corner_id:
            np.asarray(
                detection.point,
                dtype=np.float32,
            )
            for detection
            in self.detect(frame)
        }
