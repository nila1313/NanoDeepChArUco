from pathlib import Path

import yaml

from nanodeepcharuco.detection.charuco import load_board_parameters


ROOT = Path(__file__).resolve().parents[1]


def test_small_deepcharuco_config_matches_canonical_board():
    config_path = (
        ROOT
        / "configs/deepcharuco/small_5x6.yaml"
    )

    board_path = (
        ROOT
        / "configs/boards/"
          "small_5x6_dict6x6_250_meters.npy"
    )

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    board = load_board_parameters(board_path)

    assert config["board_name"] == "DICT_6X6_250"

    assert int(config["row_count"]) == int(
        board["boardHeight"]
    )

    assert int(config["col_count"]) == int(
        board["boardWidth"]
    )

    assert int(board["dictionary_type"]) == 10

    assert (
        (int(config["row_count"]) - 1)
        * (int(config["col_count"]) - 1)
    ) == 20

    deep_ratio = (
        float(config["marker_len"])
        / float(config["square_len"])
    )

    board_ratio = (
        float(board["marker_size_real"])
        / float(board["square_size_real"])
    )

    assert abs(deep_ratio - board_ratio) < 1e-9

    assert tuple(config["input_size"]) == (320, 240)
