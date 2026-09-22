from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np

from .aruco_nano import ArucoNanoDetector
from .charuco import (
    build_charuco_board,
    interpolate_charuco_from_nano,
    load_board_parameters,
)


@dataclass
class NanoCharucoDetectionResult:
    corners: dict[int, np.ndarray]
    status: str
    nano_markers: int
    charuco_corners: int


class NanoCharucoDetector:
    """
    Pure ArUco Nano baseline:

        image
          ↓
        ArUco Nano marker detection
          ↓
        OpenCV ChArUco interpolation
          ↓
        CalibCam-compatible corner dictionary

    No gamma rescue.
    No Deep ChArUco.
    No homography recovery.
    """

    def __init__(
        self,
        board_path: str | Path,
        nano_executable: str | Path,
        work_dir: str | Path,
    ):
        self.board_path = (
            Path(board_path)
            .expanduser()
            .resolve()
        )

        self.work_dir = (
            Path(work_dir)
            .expanduser()
            .resolve()
        )

        self.work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.board_params = load_board_parameters(
            self.board_path
        )

        self.dictionary_type = int(
            self.board_params["dictionary_type"]
        )

        print(
            f"Nano dictionary_type: "
            f"{self.dictionary_type}"
        )

        self.board = build_charuco_board(
            self.board_path
        )

        self.nano = ArucoNanoDetector(
            nano_executable
        )

    def process_frame(
        self,
        frame: np.ndarray,
        tag: str,
    ) -> NanoCharucoDetectionResult:

        input_dir = self.work_dir / tag

        if input_dir.exists():
            shutil.rmtree(input_dir)

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_path = (
            input_dir
            / "frame.jpg"
        )

        json_path = (
            input_dir
            / "nano.json"
        )

        ok = cv2.imwrite(
            str(image_path),
            frame,
        )

        if not ok:
            raise RuntimeError(
                f"Could not write temporary frame: "
                f"{image_path}"
            )

        result = self.nano.detect_directory(
            input_dir,
            json_path,
            dictionary_id=self.dictionary_type,
        )

        markers = result.get(
            "frame.jpg",
            [],
        )

        charuco = (
            interpolate_charuco_from_nano(
                frame,
                markers,
                self.board,
            )
        )

        corners = {
            int(c.corner_id):
            np.asarray(
                c.point,
                dtype=np.float32,
            )
            for c in charuco
        }

        return NanoCharucoDetectionResult(
            corners=corners,
            status="RAW_NANO",
            nano_markers=len(markers),
            charuco_corners=len(corners),
        )
