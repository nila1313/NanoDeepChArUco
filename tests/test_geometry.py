import numpy as np

from nanodeepcharuco.detection.geometry import board_xy, estimate_homography, spread_ok


def test_board_coordinates_support_both_boards():
    assert np.array_equal(board_xy(7, 4), np.array([3, 1], dtype=np.float32))
    assert np.array_equal(board_xy(7, 6), np.array([1, 1], dtype=np.float32))


def test_spread_uses_runtime_grid_width():
    assert spread_ok([0, 1, 6, 7], grid_width=6)
    assert spread_ok([0, 1, 4, 5], grid_width=4)
    assert not spread_ok([0, 1, 2], grid_width=4)


def test_homography_from_large_board_geometry():
    points = {idx: board_xy(idx, 6) * 10 + np.array([100, 50])
              for idx in (0, 1, 6, 7)}
    assert estimate_homography(points, grid_width=6) is not None


def test_hybrid_nano_is_good_uses_runtime_grid_width(monkeypatch):
    import numpy as np
    import nanodeepcharuco.detection.hybrid as hybrid_module
    from nanodeepcharuco.detection.hybrid import NanoDeepCharucoDetector

    monkeypatch.setattr(
        hybrid_module,
        "NANO_GOOD_MIN_CORNERS",
        4,
    )

    detector = object.__new__(
        NanoDeepCharucoDetector
    )
    detector.grid_width = 6

    nano_dict = {
        idx: np.zeros(2, dtype=float)
        for idx in [0, 1, 6, 7]
    }

    assert detector.nano_is_good(nano_dict)
