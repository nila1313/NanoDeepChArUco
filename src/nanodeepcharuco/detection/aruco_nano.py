from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

import numpy as np


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: int
    corners: np.ndarray


FrameDetections = dict[str, list[MarkerDetection]]


class ArucoNanoDetector:
    """
    Wrapper around the proven ArUco Nano `detect_batch` executable.

    Expected executable interface:
        detect_batch <image_dir> <output_json_path>

    The current C++ detector processes .jpg images and outputs:
        {
            "frame.jpg": {
                "marker_id": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            }
        }
    """

    def __init__(self, executable: str | Path):
        self.executable = Path(executable).expanduser().resolve()

        if not self.executable.is_file():
            raise FileNotFoundError(
                f"ArUco Nano executable not found: {self.executable}"
            )

    def detect_directory(
        self,
        image_dir: str | Path,
        output_json: str | Path,
        dictionary_id: int | None = None,
    ) -> FrameDetections:

        image_dir = Path(image_dir).expanduser().resolve()
        output_json = Path(output_json).expanduser().resolve()

        if not image_dir.is_dir():
            raise NotADirectoryError(
                f"Image directory not found: {image_dir}"
            )

        jpg_files = sorted(image_dir.glob("*.jpg"))

        if not jpg_files:
            raise RuntimeError(
                f"No .jpg frames found in: {image_dir}"
            )

        output_json.parent.mkdir(parents=True, exist_ok=True)

        command = [
            str(self.executable),
            str(image_dir),
            str(output_json),
        ]

        if dictionary_id is not None:
            command.append(str(int(dictionary_id)))

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "ArUco Nano failed.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        if not output_json.is_file():
            raise RuntimeError(
                f"Detector completed but output JSON was not created: "
                f"{output_json}"
            )

        return self.load_json(output_json)

    @staticmethod
    def load_json(json_path: str | Path) -> FrameDetections:
        json_path = Path(json_path)

        with json_path.open("r") as f:
            raw = json.load(f)

        detections: FrameDetections = {}

        for frame_name, marker_dict in raw.items():
            frame_detections = []

            for marker_id_str, marker_corners in marker_dict.items():
                corners = np.asarray(
                    marker_corners,
                    dtype=np.float32,
                )

                if corners.shape != (4, 2):
                    raise ValueError(
                        f"Invalid corners for marker {marker_id_str} "
                        f"in {frame_name}: shape={corners.shape}"
                    )

                frame_detections.append(
                    MarkerDetection(
                        marker_id=int(marker_id_str),
                        corners=corners,
                    )
                )

            frame_detections.sort(
                key=lambda detection: detection.marker_id
            )

            detections[frame_name] = frame_detections

        return detections
