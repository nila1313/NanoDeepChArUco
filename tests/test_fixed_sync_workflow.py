import unittest
from types import SimpleNamespace

from nanodeepcharuco.pipeline.fixed_sync import (
    find_shared_detection_ids,
)


class TestFixedSyncWorkflow(unittest.TestCase):
    def test_shared_detection_ids_are_sorted_intersection(
        self,
    ):
        left = SimpleNamespace(
            detections_by_index={
                30: {},
                10: {},
                20: {},
            },
        )

        right = SimpleNamespace(
            detections_by_index={
                40: {},
                20: {},
                10: {},
            },
        )

        self.assertEqual(
            find_shared_detection_ids(
                [left, right]
            ),
            [10, 20],
        )

    def test_no_camera_runs_returns_empty_list(
        self,
    ):
        self.assertEqual(
            find_shared_detection_ids([]),
            [],
        )


if __name__ == "__main__":
    unittest.main()
