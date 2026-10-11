from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .aruco_nano import ArucoNanoDetector
from .charuco import (
    build_charuco_board_from_params,
    interpolate_charuco_from_nano,
    load_board_parameters,
)
if TYPE_CHECKING:
    from .deep_charuco import DeepCharucoDetector

from .geometry import (
    MIN_HOMOGRAPHY_ANCHORS,
    deep_self_verify,
    fallback_recover,
    spread_ok,
)


GAMMA_DARK = 1.4
NANO_GOOD_MIN_CORNERS = 15


@dataclass
class NanoDeepCharucoDetectionResult:
    corners: dict[int, np.ndarray]
    status: str

    raw_nano: int
    gamma_nano: int | None

    nano_source: str
    nano_base: int

    deep_ran: bool = False
    deep_raw: int = 0
    deep_clean: int = 0

    duplicate_deep_ids: list[int] = field(default_factory=list)
    accepted_deep_ids: list[int] = field(default_factory=list)

    geometry_status: str = "NOT_NEEDED"


class NanoDeepCharucoDetector:
    def __init__(
        self,
        board_path: str | Path | None,
        nano_executable: str | Path,
        deep_detector: DeepCharucoDetector,
        work_dir: str | Path,
        keep_work: bool = False,
        board_params: dict | None = None,
    ):
        if board_params is None:
            if board_path is None:
                raise ValueError(
                    "Either board_path or board_params must be provided"
                )

            self.board_path = (
                Path(board_path)
                .expanduser()
                .resolve()
            )

            self.board_params = load_board_parameters(
                self.board_path
            )
        else:
            self.board_path = None
            self.board_params = dict(board_params)

        self.work_dir = (
            Path(work_dir)
            .expanduser()
            .resolve()
        )

        self.work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.keep_work = bool(keep_work)

        self.dictionary_type = int(
            self.board_params["dictionary_type"]
        )
        self.grid_width = int(self.board_params["boardWidth"]) - 1

        print(
            f"NanoDeepCharuco dictionary_type: "
            f"{self.dictionary_type}"
        )

        self.board = build_charuco_board_from_params(
            self.board_params
        )

        self.nano = ArucoNanoDetector(
            nano_executable
        )

        self.deep = deep_detector

    @staticmethod
    def gamma_dark(
        frame: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        x = gray.astype(
            np.float32
        ) / 255.0

        y = np.power(
            x,
            GAMMA_DARK,
        )

        y = np.clip(
            y * 255.0,
            0,
            255,
        ).astype(np.uint8)

        return cv2.cvtColor(
            y,
            cv2.COLOR_GRAY2BGR,
        )

    @staticmethod
    def choose_nano_base(
        raw_dict: dict[int, np.ndarray],
        gamma_dict: dict[int, np.ndarray],
    ):
        if len(gamma_dict) > len(raw_dict):
            return (
                gamma_dict,
                "GAMMA_DARK",
            )

        return (
            raw_dict,
            "RAW",
        )

    def nano_is_good(
        self,
        nano_dict: dict[int, np.ndarray],
    ) -> bool:
        ids = list(nano_dict)

        return (
            len(ids) >= NANO_GOOD_MIN_CORNERS
            and spread_ok(ids, self.grid_width)
        )

    def nano_predictions(
        self,
        frame: np.ndarray,
        tag: str,
    ):
        input_dir = self.work_dir / tag

        if input_dir.exists():
            shutil.rmtree(input_dir)

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            image_path = input_dir / "frame.jpg"
            json_path = input_dir / "nano.json"

            ok = cv2.imwrite(
                str(image_path),
                frame,
            )

            if not ok:
                raise RuntimeError(
                    f"Could not write temporary frame: {image_path}"
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

            charuco = interpolate_charuco_from_nano(
                frame,
                markers,
                self.board,
            )

            coords = {
                int(c.corner_id): np.asarray(
                    c.point,
                    dtype=np.float32,
                )
                for c in charuco
            }

            return coords

        finally:
            if not self.keep_work:
                shutil.rmtree(
                    input_dir,
                    ignore_errors=True,
                )

    def process_frame(
        self,
        frame: np.ndarray,
        tag: str,
    ) -> NanoDeepCharucoDetectionResult:

        # ============================================
        # Stage 1: RAW Nano
        # ============================================

        raw_dict = self.nano_predictions(
            frame,
            f"{tag}_raw",
        )

        if self.nano_is_good(
            raw_dict
        ):
            return NanoDeepCharucoDetectionResult(
                corners=raw_dict,
                status="RAW_NANO",

                raw_nano=len(raw_dict),
                gamma_nano=None,

                nano_source="RAW",
                nano_base=len(raw_dict),

            )

        # ============================================
        # Stage 2: Gamma-dark Nano
        # ============================================

        gamma_frame = (
            self.gamma_dark(
                frame
            )
        )

        gamma_dict = self.nano_predictions(
            gamma_frame,
            f"{tag}_gamma",
        )

        if self.nano_is_good(
            gamma_dict
        ):
            return NanoDeepCharucoDetectionResult(
                corners=gamma_dict,
                status="GAMMA_NANO",

                raw_nano=len(raw_dict),
                gamma_nano=len(gamma_dict),

                nano_source="GAMMA_DARK",
                nano_base=len(gamma_dict),

            )

        # ============================================
        # Stage 3: choose stronger Nano set
        # ============================================

        nano_base, nano_source = (
            self.choose_nano_base(
                raw_dict,
                gamma_dict,
            )
        )

        # ============================================
        # Stage 4: Deep ChArUco
        # ============================================

        deep_dict, duplicates, deep_raw = (
            self.deep.detect_with_audit(
                frame
            )
        )

        input_width = int(
            self.deep.config.input_size[0]
        )

        input_height = int(
            self.deep.config.input_size[1]
        )

        # ============================================
        # Stage 5A: Nano-homography verification
        # ============================================

        if (
            len(nano_base)
            >= MIN_HOMOGRAPHY_ANCHORS
        ):
            (
                recovered,
                accepted,
                _,
                geom_status,
            ) = fallback_recover(
                nano_base,
                deep_dict,
                frame.shape,
                input_width,
                input_height,
                self.grid_width,
            )

            return NanoDeepCharucoDetectionResult(
                corners=recovered,
                status="DEEP_NANO_H",

                raw_nano=len(raw_dict),
                gamma_nano=len(gamma_dict),

                nano_source=nano_source,
                nano_base=len(nano_base),

                deep_ran=True,
                deep_raw=deep_raw,
                deep_clean=len(deep_dict),

                duplicate_deep_ids=duplicates,
                accepted_deep_ids=accepted,

                geometry_status=geom_status,
            )

        # ============================================
        # Stage 5B: Deep self-verification
        # ============================================

        (
            verified_deep,
            _,
            geom_status,
        ) = deep_self_verify(
            deep_dict,
            frame.shape,
            input_width,
            input_height,
            self.grid_width,
        )

        recovered = {
            int(idx): np.asarray(
                xy,
                dtype=np.float32,
            )
            for idx, xy
            in nano_base.items()
        }

        accepted = []

        if geom_status == "SELF_OK":
            for idx, xy in (
                verified_deep.items()
            ):
                idx = int(idx)

                if idx in recovered:
                    continue

                recovered[idx] = (
                    np.asarray(
                        xy,
                        dtype=np.float32,
                    )
                )

                accepted.append(idx)

        return NanoDeepCharucoDetectionResult(
            corners=recovered,
            status="DEEP_SELF",

            raw_nano=len(raw_dict),
            gamma_nano=len(gamma_dict),

            nano_source=nano_source,
            nano_base=len(nano_base),

            deep_ran=True,
            deep_raw=deep_raw,
            deep_clean=len(deep_dict),

            duplicate_deep_ids=duplicates,
            accepted_deep_ids=accepted,

            geometry_status=geom_status,
        )
