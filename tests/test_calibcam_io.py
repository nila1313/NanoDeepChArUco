import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from nanodeepcharuco.calibcam_io import (
    build_calibcam_payload,
    load_calibcam_payload,
    save_calibcam_payload,
)


class TestCalibCamIO(unittest.TestCase):
    def test_build_payload_preserves_logical_and_physical_indices(self):
        detections = {
            10: {
                0: np.array(
                    [100.25, 200.5],
                    dtype=np.float32,
                ),
                2: np.array(
                    [300.0, 400.0],
                    dtype=np.float32,
                ),
            },
            20: {
                1: np.array(
                    [110.0, 210.0],
                    dtype=np.float32,
                ),
            },
        }

        physical_frames = {
            10: 11,
            20: 21,
        }

        payload = build_calibcam_payload(
            detections_by_index=detections,
            frame_indices_by_index=physical_frames,
            expected_marker_ids=[
                0,
                1,
                2,
                3,
            ],
        )

        self.assertEqual(
            payload["version"],
            "2.0",
        )
        self.assertEqual(
            payload["storage_method"],
            "array",
        )

        self.assertEqual(
            payload["marker_ids"],
            [0, 1, 2, 3],
        )

        self.assertEqual(
            payload["detection_idxs"],
            [10, 20],
        )

        self.assertEqual(
            payload["frame_idxs"],
            [[11, 21]],
        )

        coords = np.asarray(
            payload["marker_coords"],
            dtype=np.float32,
        )

        self.assertEqual(
            coords.shape,
            (1, 2, 4, 2),
        )

        np.testing.assert_allclose(
            coords[0, 0, 0],
            [100.25, 200.5],
        )

        np.testing.assert_allclose(
            coords[0, 0, 2],
            [300.0, 400.0],
        )

        np.testing.assert_allclose(
            coords[0, 1, 1],
            [110.0, 210.0],
        )

        self.assertTrue(
            np.isnan(
                coords[0, 0, 1]
            ).all()
        )

        self.assertTrue(
            np.isnan(
                coords[:, :, 3]
            ).all()
        )

    def test_yaml_round_trip_preserves_calibcam_payload(self):
        payload = build_calibcam_payload(
            detections_by_index={
                5: {
                    0: np.array(
                        [12.5, 22.5],
                        dtype=np.float32,
                    ),
                },
            },
            frame_indices_by_index={
                5: 7,
            },
            expected_marker_ids=[
                0,
                1,
            ],
        )

        with TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "detection_000.yml"
            )

            saved = save_calibcam_payload(
                path,
                payload,
            )

            loaded = load_calibcam_payload(
                saved
            )

        self.assertEqual(
            loaded["version"],
            "2.0",
        )
        self.assertEqual(
            loaded["storage_method"],
            "array",
        )
        self.assertEqual(
            loaded["marker_ids"],
            [0, 1],
        )
        self.assertEqual(
            loaded["detection_idxs"],
            [5],
        )
        self.assertEqual(
            loaded["frame_idxs"],
            [[7]],
        )

        coords = np.asarray(
            loaded["marker_coords"],
            dtype=np.float32,
        )

        self.assertEqual(
            coords.shape,
            (1, 1, 2, 2),
        )

        np.testing.assert_allclose(
            coords[0, 0, 0],
            [12.5, 22.5],
        )

        self.assertTrue(
            np.isnan(
                coords[0, 0, 1]
            ).all()
        )

    def test_unexpected_marker_id_is_rejected(self):
        with self.assertRaises(ValueError):
            build_calibcam_payload(
                detections_by_index={
                    10: {
                        5: np.array(
                            [1.0, 2.0],
                            dtype=np.float32,
                        ),
                    },
                },
                frame_indices_by_index={
                    10: 10,
                },
                expected_marker_ids=[
                    0,
                    1,
                    2,
                ],
            )


if __name__ == "__main__":
    unittest.main()
