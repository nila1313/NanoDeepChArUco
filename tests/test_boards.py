from pathlib import Path
import hashlib
import numpy as np
from nanodeepcharuco.detection.charuco import (
    build_charuco_board_from_params,
    load_board_parameters,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_board_definitions_are_canonical():
    large = ROOT / "configs/boards/large_7x7_dict4x4_50.npy"
    small = ROOT / "configs/boards/small_5x6_dict6x6_250_meters.npy"
    assert sha256(large) == "be24cd8e2c76f99b087d7e9701a9c31988a69fcd28e392ce957ea7842fe2cc24"
    assert sha256(small) == "69ef6a2f5708b978483b278d43b374a3576f908b5adfb71da8afdfab8e684ff5"
    assert load_board_parameters(large)["dictionary_type"] == 0
    assert load_board_parameters(small)["dictionary_type"] == 10

def test_build_charuco_board_from_params():
    small = ROOT / "configs/boards/small_5x6_dict6x6_250_meters.npy"
    params = load_board_parameters(small)

    board = build_charuco_board_from_params(params)

    assert tuple(board.getChessboardSize()) == (
        params["boardWidth"],
        params["boardHeight"],
    )
    assert np.isclose(board.getSquareLength(), params["square_size_real"])
    assert np.isclose(board.getMarkerLength(), params["marker_size_real"])
