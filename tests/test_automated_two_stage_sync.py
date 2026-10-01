import numpy as np

from nanodeepcharuco.calibcam.payload import (
    build_calibcam_dict_from_selected_frames,
)
from nanodeepcharuco.sync.epipolar import (
    EpipolarOffsetScore,
)
from nanodeepcharuco.sync.pairing import (
    build_synchronized_pairs,
)
from nanodeepcharuco.sync.schedule import (
    build_auto_sync_detection_maps,
)
from nanodeepcharuco.sync.segments import (
    SyncSegment,
)
from nanodeepcharuco.sync.windowed import (
    WindowSyncResult,
)


def test_auto_sync_schedule_expands_right_candidate_frames():
    left_map, right_map = build_auto_sync_detection_maps(
        frames_start=20,
        frames_end=61,
        frames_step=20,
        offsets=[-1, 0, 1],
        left_frame_count=100,
        right_frame_count=100,
    )

    assert list(left_map) == [
        20,
        40,
        60,
    ]

    assert list(right_map) == [
        19,
        20,
        21,
        39,
        40,
        41,
        59,
        60,
        61,
    ]


def test_trusted_segment_builds_right_plus_one_pairs():
    score = EpipolarOffsetScore(
        offset=1,
        n_frame_pairs=10,
        n_correspondences=100,
        n_inliers=80,
        inlier_ratio=0.80,
        median_sampson_error=0.1,
        mean_sampson_error=0.2,
    )

    window = WindowSyncResult(
        start_frame=0,
        end_frame=100,
        best_offset=1,
        best_score=score,
        second_score=None,
        inlier_ratio_gap=0.80,
        n_left_frames=100,
        n_right_frames=100,
    )

    segment = SyncSegment(
        start_frame=0,
        end_frame=100,
        offset=1,
        n_support_windows=2,
        mean_inlier_ratio=0.80,
        mean_ratio_gap=0.80,
    )

    pairs = build_synchronized_pairs(
        left_frames=[
            0,
            20,
            40,
        ],
        right_frames=[
            1,
            21,
            41,
        ],
        segments=[segment],
        require_right_available=True,
        window_results=[window],
        min_inlier_ratio=0.20,
        min_ratio_gap=0.05,
    )

    assert [
        (
            pair.left_frame,
            pair.right_frame,
            pair.offset,
        )
        for pair in pairs
    ] == [
        (0, 1, 1),
        (20, 21, 1),
        (40, 41, 1),
    ]


def test_selected_payload_keeps_logical_and_physical_indices_separate():
    left_data = {
        400: {
            0: np.array(
                [10.0, 20.0],
                dtype=np.float32,
            ),
        },
        420: {
            0: np.array(
                [11.0, 21.0],
                dtype=np.float32,
            ),
        },
    }

    right_data = {
        401: {
            0: np.array(
                [30.0, 40.0],
                dtype=np.float32,
            ),
        },
        421: {
            0: np.array(
                [31.0, 41.0],
                dtype=np.float32,
            ),
        },
    }

    logical_ids = [
        0,
        1,
    ]

    left_payload = (
        build_calibcam_dict_from_selected_frames(
            left_data,
            [400, 420],
            logical_ids,
        )
    )

    right_payload = (
        build_calibcam_dict_from_selected_frames(
            right_data,
            [401, 421],
            logical_ids,
        )
    )

    assert left_payload["detection_idxs"] == [
        0,
        1,
    ]

    assert right_payload["detection_idxs"] == [
        0,
        1,
    ]

    assert left_payload["frame_idxs"] == [[
        400,
        420,
    ]]

    assert right_payload["frame_idxs"] == [[
        401,
        421,
    ]]
