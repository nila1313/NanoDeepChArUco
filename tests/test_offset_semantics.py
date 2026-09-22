import numpy as np

from nanodeepcharuco.calibcam.payload import (
    build_calibcam_dict,
)


def test_positive_offset_uses_common_detection_slot():
    side_data = {
        661: {
            0: np.array(
                [100.0, 200.0],
                dtype=np.float32,
            ),
        }
    }

    payload = build_calibcam_dict(
        side_data,
        frame_offset=1,
        frames_start=660,
        frames_step=20,
    )

    # Stereo sample slot is common.
    assert payload["detection_idxs"] == [0]

    # Physical right-camera video frame is preserved.
    assert payload["frame_idxs"] == [[661]]


def test_negative_offset_uses_common_detection_slot():
    side_data = {
        657: {
            0: np.array(
                [100.0, 200.0],
                dtype=np.float32,
            ),
        }
    }

    payload = build_calibcam_dict(
        side_data,
        frame_offset=-3,
        frames_start=660,
        frames_step=20,
    )

    # Physical 657 with offset -3 corresponds
    # to aligned stereo frame 660 -> sample slot 0.
    assert payload["detection_idxs"] == [0]
    assert payload["frame_idxs"] == [[657]]
