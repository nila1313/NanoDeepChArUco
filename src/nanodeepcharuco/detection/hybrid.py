from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from .nano import ArucoNanoDetector
from .charuco import (
    build_charuco_board,
    interpolate_charuco_from_nano,
    load_board_parameters,
)
from .gamma import gamma_dark
from .deep import DeepCharucoDetector
from .geometry import (
    MIN_HOMOGRAPHY_ANCHORS,
    deep_self_verify,
    fallback_recover,
    spread_ok,
)


GAMMA_DARK = 1.4
NANO_GOOD_MIN_CORNERS = 15


@dataclass
class HybridDetectionResult:
    corners: dict[int, np.ndarray]
    status: str

    raw_nano: int
    gamma_nano: int | None

    nano_source: str
    nano_base: int

    deep_ran: bool
    deep_raw: int
    deep_clean: int

    duplicate_deep_ids: list[int]
    accepted_deep_ids: list[int]

    geometry_status: str


class NanoDeepCharucoDetector:
    def __init__(
        self,
        board_path: str | Path,
        nano_executable: str | Path,
        deep_detector: DeepCharucoDetector,
        work_dir: str | Path,
        gamma: float = GAMMA_DARK,
        deep_self_ransac_px: float = 5.0,
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

        self.board_params = (
            load_board_parameters(
                self.board_path
            )
        )

        self.board_width = int(
            self.board_params["boardWidth"]
        )

        self.board_height = int(
            self.board_params["boardHeight"]
        )

        self.dictionary_type = int(
            self.board_params[
                "dictionary_type"
            ]
        )

        self.gamma = float(
            gamma
        )

        self.deep_self_ransac_px = float(
            deep_self_ransac_px
        )

        self.board = build_charuco_board(
            self.board_path
        )

        self.nano = ArucoNanoDetector(
            nano_executable
        )

        self.deep = deep_detector

        self._validate_deep_board_compatibility()

    def _validate_deep_board_compatibility(
        self,
    ) -> None:
        if self.deep.config is None:
            raise RuntimeError(
                "DeepCharucoDetector is not loaded. "
                "Call .load() before constructing "
                "NanoDeepCharucoDetector."
            )

        deep_rows = int(
            self.deep.config.row_count
        )

        deep_cols = int(
            self.deep.config.col_count
        )

        deep_n_ids = int(
            self.deep.config.n_ids
        )

        expected_n_ids = (
            (self.board_width - 1)
            * (self.board_height - 1)
        )

        if (
            deep_cols != self.board_width
            or deep_rows != self.board_height
        ):
            raise ValueError(
                "DeepChArUco config does not match board: "
                f"board={self.board_width}x{self.board_height}, "
                f"deep={deep_cols}x{deep_rows}"
            )

        if deep_n_ids != expected_n_ids:
            raise ValueError(
                "DeepChArUco corner-ID count does not "
                "match board: "
                f"expected={expected_n_ids}, "
                f"deep={deep_n_ids}"
            )

    def nano_is_good(
        self,
        nano_dict: dict[int, np.ndarray],
    ) -> bool:
        ids = list(
            nano_dict
        )

        return (
            len(ids)
            >= NANO_GOOD_MIN_CORNERS
            and spread_ok(
                ids,
                board_width=self.board_width,
            )
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

    def nano_predictions(
        self,
        frame: np.ndarray,
        tag: str,
    ):
        self.work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with tempfile.TemporaryDirectory(
            prefix=f"{tag}_",
            dir=self.work_dir,
        ) as temp_dir:
            input_dir = Path(
                temp_dir
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
                    "Could not write temporary frame: "
                    f"{image_path}"
                )

            result = self.nano.detect_directory(
                input_dir,
                json_path,
                dictionary_id=(
                    self.dictionary_type
                ),
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

            coords = {
                int(c.corner_id):
                np.asarray(
                    c.point,
                    dtype=np.float32,
                )
                for c in charuco
            }

            marker_count = len(
                markers
            )

        return (
            coords,
            marker_count,
        )


    def deep_predictions(
        self,
        frame: np.ndarray,
    ):
        return (
            self.deep.detect_with_audit(
                frame
            )
        )

    def process_frame(
        self,
        frame: np.ndarray,
        tag: str,
    ) -> HybridDetectionResult:

        # Stage 1: RAW Nano
        raw_dict, _ = (
            self.nano_predictions(
                frame,
                f"{tag}_raw",
            )
        )

        if self.nano_is_good(
            raw_dict
        ):
            return HybridDetectionResult(
                corners=raw_dict,
                status="RAW_NANO",

                raw_nano=len(
                    raw_dict
                ),
                gamma_nano=None,

                nano_source="RAW",
                nano_base=len(
                    raw_dict
                ),

                deep_ran=False,
                deep_raw=0,
                deep_clean=0,

                duplicate_deep_ids=[],
                accepted_deep_ids=[],

                geometry_status=(
                    "NOT_NEEDED"
                ),
            )

        # Stage 2: Gamma-dark Nano
        gamma_frame = gamma_dark(
            frame,
            gamma=self.gamma,
        )

        gamma_dict, _ = (
            self.nano_predictions(
                gamma_frame,
                f"{tag}_gamma",
            )
        )

        if self.nano_is_good(
            gamma_dict
        ):
            return HybridDetectionResult(
                corners=gamma_dict,
                status="GAMMA_NANO",

                raw_nano=len(
                    raw_dict
                ),
                gamma_nano=len(
                    gamma_dict
                ),

                nano_source=(
                    "GAMMA_DARK"
                ),
                nano_base=len(
                    gamma_dict
                ),

                deep_ran=False,
                deep_raw=0,
                deep_clean=0,

                duplicate_deep_ids=[],
                accepted_deep_ids=[],

                geometry_status=(
                    "NOT_NEEDED"
                ),
            )

        # Stage 3: choose stronger Nano set
        nano_base, nano_source = (
            self.choose_nano_base(
                raw_dict,
                gamma_dict,
            )
        )

        # Stage 4: Deep ChArUco
        (
            deep_dict,
            duplicates,
            deep_raw,
        ) = self.deep_predictions(
            frame
        )

        input_width = int(
            self.deep.config.input_size[0]
        )

        input_height = int(
            self.deep.config.input_size[1]
        )

        # Stage 5A: Nano-homography verification
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
                nano_dict=nano_base,
                deep_dict=deep_dict,
                frame_shape=frame.shape,
                deep_input_width=input_width,
                deep_input_height=input_height,
                board_width=(
                    self.board_width
                ),
            )

            return HybridDetectionResult(
                corners=recovered,
                status="DEEP_NANO_H",

                raw_nano=len(
                    raw_dict
                ),
                gamma_nano=len(
                    gamma_dict
                ),

                nano_source=nano_source,
                nano_base=len(
                    nano_base
                ),

                deep_ran=True,
                deep_raw=deep_raw,
                deep_clean=len(
                    deep_dict
                ),

                duplicate_deep_ids=(
                    duplicates
                ),
                accepted_deep_ids=(
                    accepted
                ),

                geometry_status=(
                    geom_status
                ),
            )

        # Stage 5B: Deep self-verification
        (
            verified_deep,
            _,
            geom_status,
        ) = deep_self_verify(
            deep_dict=deep_dict,
            frame_shape=frame.shape,
            deep_input_width=input_width,
            deep_input_height=input_height,
            board_width=(
                self.board_width
            ),
            ransac_eq_px=(
                self.deep_self_ransac_px
            ),
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
            for (
                idx,
                xy,
            ) in verified_deep.items():

                idx = int(
                    idx
                )

                if idx in recovered:
                    continue

                recovered[
                    idx
                ] = np.asarray(
                    xy,
                    dtype=np.float32,
                )

                accepted.append(
                    idx
                )

        return HybridDetectionResult(
            corners=recovered,
            status="DEEP_SELF",

            raw_nano=len(
                raw_dict
            ),
            gamma_nano=len(
                gamma_dict
            ),

            nano_source=nano_source,
            nano_base=len(
                nano_base
            ),

            deep_ran=True,
            deep_raw=deep_raw,
            deep_clean=len(
                deep_dict
            ),

            duplicate_deep_ids=(
                duplicates
            ),
            accepted_deep_ids=(
                accepted
            ),

            geometry_status=(
                geom_status
            ),
        )
