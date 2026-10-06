import numpy as np

from nanodeepcharuco.calibcam.outlier_rejection import (
    _subset_detection_payload,
    analyze_board_positions,
)


def test_outlier_analysis_rejects_single_bad_pair(
    tmp_path,
):
    import yaml

    frame_idxs = [
        [100, 200, 300],
        [101, 201, 301],
    ]

    residuals = np.ones(
        (2, 3, 2, 2),
        dtype=float,
    )

    # One very bad left-camera corner.
    residuals[0, 1, 0] = [
        20.0,
        1.0,
    ]

    data = {
        "frame_idxs": frame_idxs,
        "info": {
            "fun_final": (
                residuals
                .reshape(-1)
                .tolist()
            ),
        },
        "rvecs": [],
        "tvecs": [],
        "version": 4,
    }

    path = (
        tmp_path
        / "multicam_calibration_board_positions.yml"
    )

    path.write_text(
        yaml.safe_dump(data)
    )

    result = analyze_board_positions(
        path,
        mad_multiplier=6.0,
        min_threshold_px=6.0,
        max_threshold_px=10.0,
    )

    assert result["rejected_indices"] == [1]
    assert result["pairs"][1]["left_frame"] == 200
    assert result["pairs"][1]["right_frame"] == 201


def test_detection_payload_subset_keeps_alignment():
    payload = {
        "version": "1.0",
        "storage_method": "numpy",
        "marker_coords": [[
            "a",
            "b",
            "c",
        ]],
        "marker_ids": list(range(20)),
        "detection_idxs": [
            10,
            11,
            12,
        ],
        "frame_idxs": [[
            100,
            200,
            300,
        ]],
    }

    result = _subset_detection_payload(
        payload,
        [0, 2],
    )

    assert result["marker_coords"] == [[
        "a",
        "c",
    ]]

    assert result["detection_idxs"] == [
        10,
        12,
    ]

    assert result["frame_idxs"] == [[
        100,
        300,
    ]]

    # Global marker IDs stay unchanged.
    assert result["marker_ids"] == list(
        range(20)
    )
