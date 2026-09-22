import numpy as np

from nanodeepcharuco.calibcam.payload import build_calibcam_dict


def test_payload_schema_and_offset():
    # side_data keys are physical video-frame indices.
    # With offset +1:
    #   physical 1  -> aligned 0
    #   physical 21 -> aligned 20
    side_data = {
        1: {1: np.array([10, 20], dtype=np.float32)},
        21: {3: np.array([30, 40], dtype=np.float32)},
    }
    payload = build_calibcam_dict(side_data, frame_offset=1)
    assert payload["version"] == "2.0"
    assert payload["storage_method"] == "array"
    assert payload["marker_ids"] == [1, 3]
    assert payload["detection_idxs"] == [0, 20]
    assert payload["frame_idxs"] == [[1, 21]]
    coords = np.asarray(payload["marker_coords"], dtype=np.float32)
    assert coords.shape == (1, 2, 2, 2)
    assert np.isnan(coords[0, 0, 1]).all()


def test_empty_payload_is_valid():
    payload = build_calibcam_dict({})
    assert payload["marker_ids"] == []
    assert payload["detection_idxs"] == []
    assert payload["frame_idxs"] == [[]]
