from types import SimpleNamespace

import numpy as np

from nanodeepcharuco.sync.stability import (
    find_stable_board_segments,
    filter_pairs_by_board_stability,
)


def corners(x_shift):
    return {
        0: np.array([0.0 + x_shift, 0.0]),
        1: np.array([10.0 + x_shift, 0.0]),
        2: np.array([0.0 + x_shift, 10.0]),
        3: np.array([10.0 + x_shift, 10.0]),
    }


def test_detects_consecutive_stable_board_frames():
    detections = {
        10: corners(0.0),
        11: corners(0.5),
        12: corners(0.8),
        13: corners(1.1),
        14: corners(10.0),
    }

    segments = find_stable_board_segments(
        detections,
        motion_threshold_px=2.0,
        min_stable_frames=3,
    )

    assert len(segments) == 1

    segment = segments[0]

    assert segment.start_frame == 10
    assert segment.end_frame == 14
    assert segment.n_frames == 4


def test_large_board_motion_is_not_stable():
    detections = {
        10: corners(0.0),
        11: corners(5.0),
        12: corners(10.0),
        13: corners(15.0),
    }

    segments = find_stable_board_segments(
        detections,
        motion_threshold_px=2.0,
        min_stable_frames=3,
    )

    assert segments == []


def test_pair_requires_both_cameras_to_be_stable():
    pair_a = SimpleNamespace(
        left_frame=10,
        right_frame=11,
    )

    pair_b = SimpleNamespace(
        left_frame=30,
        right_frame=31,
    )

    left_segments = [
        SimpleNamespace(
            start_frame=10,
            end_frame=20,
        ),
        SimpleNamespace(
            start_frame=30,
            end_frame=40,
        ),
    ]

    right_segments = [
        SimpleNamespace(
            start_frame=11,
            end_frame=20,
        ),
    ]

    selected = filter_pairs_by_board_stability(
        [pair_a, pair_b],
        left_segments,
        right_segments,
    )

    assert selected == [pair_a]


def test_fixed_motion_threshold_is_preserved():
    from nanodeepcharuco.sync.stability import (
        select_motion_threshold,
    )

    threshold = select_motion_threshold(
        {},
        mode="fixed",
        fixed_threshold_px=2.0,
        percentile=25.0,
        max_threshold_px=6.0,
    )

    assert threshold == 2.0


def test_adaptive_motion_threshold_respects_floor(
    monkeypatch,
):
    import nanodeepcharuco.sync.stability as stability

    monkeypatch.setattr(
        stability,
        "build_motion_signal",
        lambda *args, **kwargs: {
            1: 0.2,
            2: 0.3,
            3: 0.4,
            4: 0.5,
        },
    )

    threshold = stability.select_motion_threshold(
        {},
        mode="adaptive",
        fixed_threshold_px=2.0,
        percentile=25.0,
        max_threshold_px=6.0,
    )

    assert threshold == 2.0


def test_adaptive_motion_threshold_uses_percentile(
    monkeypatch,
):
    import nanodeepcharuco.sync.stability as stability

    monkeypatch.setattr(
        stability,
        "build_motion_signal",
        lambda *args, **kwargs: {
            1: 2.0,
            2: 3.0,
            3: 4.0,
            4: 5.0,
            5: 6.0,
        },
    )

    threshold = stability.select_motion_threshold(
        {},
        mode="adaptive",
        fixed_threshold_px=2.0,
        percentile=50.0,
        max_threshold_px=6.0,
    )

    assert threshold == 4.0


def test_adaptive_motion_threshold_respects_cap(
    monkeypatch,
):
    import nanodeepcharuco.sync.stability as stability

    monkeypatch.setattr(
        stability,
        "build_motion_signal",
        lambda *args, **kwargs: {
            1: 8.0,
            2: 9.0,
            3: 10.0,
            4: 11.0,
        },
    )

    threshold = stability.select_motion_threshold(
        {},
        mode="adaptive",
        fixed_threshold_px=2.0,
        percentile=50.0,
        max_threshold_px=6.0,
    )

    assert threshold == 6.0
