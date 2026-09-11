from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BoardInfo:
    path: str
    raw: object


def load_board(path: str) -> BoardInfo:
    board_path = Path(path)

    if not board_path.is_file():
        raise FileNotFoundError(
            f"Board file not found: {board_path}"
        )

    if board_path.suffix.lower() != ".npy":
        raise ValueError(
            "Current NanoDeepChArUco board loader "
            "expects a .npy board file."
        )

    data = np.load(
        board_path,
        allow_pickle=True,
    )

    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()

    return BoardInfo(
        path=str(board_path.resolve()),
        raw=data,
    )


def describe_board(
    board: BoardInfo,
) -> dict:
    raw = board.raw

    summary = {
        "path": board.path,
        "type": type(raw).__name__,
    }

    if isinstance(raw, dict):
        summary["keys"] = list(raw.keys())

        for key in [
            "boardWidth",
            "boardHeight",
            "squareLength",
            "markerLength",
            "dictionary",
            "aruco_dict",
            "dict",
        ]:
            if key in raw:
                value = raw[key]

                if isinstance(value, np.ndarray):
                    value = value.tolist()

                summary[key] = value

    elif isinstance(raw, np.ndarray):
        summary["shape"] = list(raw.shape)
        summary["dtype"] = str(raw.dtype)

    return summary


@dataclass(frozen=True)
class NormalizedBoard:
    board_width: int
    board_height: int
    square_size: float
    marker_size: float
    square_size_real: float
    marker_size_real: float
    dictionary_type: int
    charuco_corners: int


def normalize_board(
    board: BoardInfo,
) -> NormalizedBoard:
    raw = board.raw

    if not isinstance(raw, dict):
        raise TypeError(
            "Expected board data to be a dictionary."
        )

    required = [
        "boardWidth",
        "boardHeight",
        "square_size",
        "marker_size",
        "square_size_real",
        "marker_size_real",
        "dictionary_type",
    ]

    missing = [
        key
        for key in required
        if key not in raw
    ]

    if missing:
        raise KeyError(
            f"Board file missing required keys: {missing}"
        )

    board_width = int(
        raw["boardWidth"]
    )

    board_height = int(
        raw["boardHeight"]
    )

    charuco_corners = (
        (board_width - 1)
        * (board_height - 1)
    )

    return NormalizedBoard(
        board_width=board_width,
        board_height=board_height,
        square_size=float(
            raw["square_size"]
        ),
        marker_size=float(
            raw["marker_size"]
        ),
        square_size_real=float(
            raw["square_size_real"]
        ),
        marker_size_real=float(
            raw["marker_size_real"]
        ),
        dictionary_type=int(
            raw["dictionary_type"]
        ),
        charuco_corners=charuco_corners,
    )
