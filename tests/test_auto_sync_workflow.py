import unittest
from types import SimpleNamespace

from nanodeepcharuco.pipeline.auto_sync import (
    run_to_physical_detections,
)


class TestAutoSyncWorkflow(unittest.TestCase):
    def test_physical_detection_mapping_uses_frame_indices(
        self,
    ):
        corners_a = {
            1: "corner-a",
        }

        corners_b = {
            2: "corner-b",
        }

        run = SimpleNamespace(
            detections_by_index={
                10: corners_a,
                20: corners_b,
            },
            frame_indices_by_index={
                10: 101,
                20: 205,
            },
        )

        self.assertEqual(
            run_to_physical_detections(run),
            {
                101: corners_a,
                205: corners_b,
            },
        )

    def test_detection_without_physical_frame_is_skipped(
        self,
    ):
        run = SimpleNamespace(
            detections_by_index={
                10: {
                    1: "corner",
                },
            },
            frame_indices_by_index={},
        )

        self.assertEqual(
            run_to_physical_detections(run),
            {},
        )


if __name__ == "__main__":
    unittest.main()
