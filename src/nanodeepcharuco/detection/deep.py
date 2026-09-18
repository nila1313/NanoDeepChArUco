from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class DeepCharucoDetection:
    corner_id: int
    point: np.ndarray


class DeepCharucoDetector:
    """
    Adapter around the verified DeepChArUco inference code.

    All external paths are provided explicitly so the package
    remains portable across machines.
    """

    def __init__(
        self,
        deepcharuco_root: str | Path,
        deep_checkpoint: str | Path,
        refinenet_checkpoint: str | Path,
        config_path: str | Path,
        device: str | None = None,
    ):
        self.deepcharuco_root = (
            Path(deepcharuco_root)
            .expanduser()
            .resolve()
        )

        self.models_src = (
            self.deepcharuco_root
            / "models"
        )

        if not self.deepcharuco_root.is_dir():
            raise NotADirectoryError(
                "DeepChArUco source directory not found: "
                f"{self.deepcharuco_root}"
            )

        if str(self.deepcharuco_root) not in sys.path:
            sys.path.insert(
                0,
                str(self.deepcharuco_root),
            )

        if self.models_src.is_dir():
            if str(self.models_src) not in sys.path:
                sys.path.insert(
                    0,
                    str(self.models_src),
                )

        from configs import load_configuration
        from inference import load_models, infer_image

        self._load_configuration = (
            load_configuration
        )

        self._load_models = (
            load_models
        )

        self._infer_image = (
            infer_image
        )

        self.deep_checkpoint = (
            Path(deep_checkpoint)
            .expanduser()
            .resolve()
        )

        self.refinenet_checkpoint = (
            Path(refinenet_checkpoint)
            .expanduser()
            .resolve()
        )

        self.config_path = (
            Path(config_path)
            .expanduser()
            .resolve()
        )

        if not self.deep_checkpoint.is_file():
            raise FileNotFoundError(
                "DeepChArUco checkpoint not found: "
                f"{self.deep_checkpoint}"
            )

        if not self.refinenet_checkpoint.is_file():
            raise FileNotFoundError(
                "RefineNet checkpoint not found: "
                f"{self.refinenet_checkpoint}"
            )

        if not self.config_path.is_file():
            raise FileNotFoundError(
                "DeepChArUco config not found: "
                f"{self.config_path}"
            )

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(
            device
        )

        self.config = None
        self.deepc = None
        self.refinenet = None

    def load(self):
        self.config = (
            self._load_configuration(
                str(self.config_path)
            )
        )

        (
            self.deepc,
            self.refinenet,
        ) = self._load_models(
            str(self.deep_checkpoint),
            str(self.refinenet_checkpoint),
            n_ids=self.config.n_ids,
            device=self.device,
        )

        return self

    def _infer_keypoints(
        self,
        frame: np.ndarray,
    ):
        if self.config is None:
            raise RuntimeError(
                "DeepCharucoDetector is not loaded. "
                "Call .load() first."
            )

        if frame is None:
            raise ValueError(
                "Input frame is None"
            )

        h, w = frame.shape[:2]

        input_width = int(
            self.config.input_size[0]
        )

        input_height = int(
            self.config.input_size[1]
        )

        small = cv2.resize(
            frame,
            (
                input_width,
                input_height,
            ),
            interpolation=cv2.INTER_LINEAR_EXACT,
        )

        keypoints, _ = self._infer_image(
            small,
            self.config.n_ids,
            self.deepc,
            self.refinenet,
            draw_pred=False,
            device=self.device,
        )

        scale_x = (
            w
            / float(input_width)
        )

        scale_y = (
            h
            / float(input_height)
        )

        return (
            keypoints,
            scale_x,
            scale_y,
        )

    def detect_with_audit(
        self,
        frame: np.ndarray,
    ):
        """
        Return:
            clean detections,
            duplicate corner IDs,
            raw Deep prediction count.
        """

        (
            keypoints,
            scale_x,
            scale_y,
        ) = self._infer_keypoints(
            frame
        )

        by_id = {}

        for x, y, corner_id in keypoints:
            corner_id = int(
                corner_id
            )

            by_id.setdefault(
                corner_id,
                [],
            ).append(
                np.asarray(
                    [
                        float(x) * scale_x,
                        float(y) * scale_y,
                    ],
                    dtype=np.float32,
                )
            )

        clean = {}
        duplicates = []

        for (
            corner_id,
            points,
        ) in by_id.items():

            if len(points) == 1:
                clean[
                    corner_id
                ] = points[0]

            else:
                duplicates.append(
                    corner_id
                )

        return (
            clean,
            sorted(
                duplicates
            ),
            len(
                keypoints
            ),
        )

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[DeepCharucoDetection]:

        clean, _, _ = (
            self.detect_with_audit(
                frame
            )
        )

        detections = [
            DeepCharucoDetection(
                corner_id=int(
                    corner_id
                ),
                point=np.asarray(
                    point,
                    dtype=np.float32,
                ),
            )
            for (
                corner_id,
                point,
            ) in clean.items()
        ]

        detections.sort(
            key=lambda detection:
            detection.corner_id
        )

        return detections
