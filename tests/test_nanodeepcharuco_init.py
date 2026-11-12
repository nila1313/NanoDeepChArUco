from pathlib import Path

from nanodeepcharuco.detection.charuco import load_board_parameters
from nanodeepcharuco.detection.nanodeepcharuco import NanoDeepCharucoDetector


ROOT = Path(__file__).resolve().parents[1]


def test_nanodeepcharuco_accepts_board_path_and_board_params(tmp_path):
    board_path = (
        ROOT
        / "configs/boards/small_5x6_dict6x6_250_meters.npy"
    )
    board_params = load_board_parameters(board_path)

    # ArucoNanoDetector only requires the executable path to exist
    # during construction. It is not executed in this test.
    fake_nano = tmp_path / "detect_batch"
    fake_nano.touch()

    # The NanoDeepCharuco detector only stores the Deep detector at init time.
    fake_deep = object()

    detector_from_path = NanoDeepCharucoDetector(
        board_path,
        fake_nano,
        fake_deep,
        tmp_path / "work_path",
    )

    detector_from_params = NanoDeepCharucoDetector(
        None,
        fake_nano,
        fake_deep,
        tmp_path / "work_params",
        board_params=board_params,
    )

    assert detector_from_path.board_path == board_path.resolve()
    assert detector_from_params.board_path is None

    assert detector_from_path.dictionary_type == 10
    assert detector_from_params.dictionary_type == 10

    assert (
        detector_from_path.board_params["boardWidth"]
        == detector_from_params.board_params["boardWidth"]
    )
    assert (
        detector_from_path.board_params["boardHeight"]
        == detector_from_params.board_params["boardHeight"]
    )

    assert tuple(detector_from_path.board.getChessboardSize()) == \
        tuple(detector_from_params.board.getChessboardSize())
