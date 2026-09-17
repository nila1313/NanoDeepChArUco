import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from nanodeepcharuco.detection.charuco import (
    expected_charuco_corner_ids,
)


class TestCharucoBoard(unittest.TestCase):
    def _write_board(
        self,
        path: Path,
        width: int,
        height: int,
    ) -> None:
        np.save(
            path,
            {
                "boardWidth": width,
                "boardHeight": height,
                "square_size_real": 0.02,
                "marker_size_real": 0.013333,
                "dictionary_type": 10,
            },
            allow_pickle=True,
        )

    def test_small_board_has_twenty_charuco_ids(self):
        with TemporaryDirectory() as tmp:
            board = Path(tmp) / "small.npy"

            self._write_board(
                board,
                width=5,
                height=6,
            )

            self.assertEqual(
                expected_charuco_corner_ids(
                    board
                ),
                list(range(20)),
            )

    def test_large_board_has_thirty_six_charuco_ids(self):
        with TemporaryDirectory() as tmp:
            board = Path(tmp) / "large.npy"

            self._write_board(
                board,
                width=7,
                height=7,
            )

            self.assertEqual(
                expected_charuco_corner_ids(
                    board
                ),
                list(range(36)),
            )


if __name__ == "__main__":
    unittest.main()
