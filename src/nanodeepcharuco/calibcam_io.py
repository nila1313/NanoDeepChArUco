from __future__ import annotations

from pathlib import Path

import numpy as np


def build_calibcam_payload(
    detections_by_index: dict[
        int,
        dict[int, np.ndarray],
    ],
    frame_indices_by_index: dict[
        int,
        int,
    ],
    expected_marker_ids: list[int] | None = None,
) -> dict:
    """
    Build a CalibCam-compatible single-camera detection payload.

    detection index:
        shared logical pairing index used to associate cameras

    frame index:
        actual physical video frame used for this camera
    """

    detection_idxs = sorted(
        int(idx)
        for idx in detections_by_index
    )

    if not detection_idxs:
        raise ValueError(
            "No detections were provided."
        )

    missing_frames = [
        idx
        for idx in detection_idxs
        if idx not in frame_indices_by_index
    ]

    if missing_frames:
        raise ValueError(
            "Missing physical frame indices for "
            f"detection indices: {missing_frames}"
        )

    observed_marker_ids = sorted({
        int(marker_id)
        for detection_idx in detection_idxs
        for marker_id
        in detections_by_index[
            detection_idx
        ]
    })

    if not observed_marker_ids:
        raise ValueError(
            "No ChArUco marker IDs were detected."
        )

    if expected_marker_ids is None:
        marker_ids = observed_marker_ids
    else:
        marker_ids = sorted({
            int(marker_id)
            for marker_id
            in expected_marker_ids
        })

        unexpected = sorted(
            set(observed_marker_ids)
            - set(marker_ids)
        )

        if unexpected:
            raise ValueError(
                "Detected ChArUco IDs are outside "
                "the expected board ID range: "
                f"{unexpected}"
            )

    id_to_col = {
        marker_id: col
        for col, marker_id
        in enumerate(marker_ids)
    }

    n_frames = len(
        detection_idxs
    )

    n_markers = len(
        marker_ids
    )

    coords = np.full(
        (
            n_frames,
            n_markers,
            2,
        ),
        np.nan,
        dtype=np.float32,
    )

    frame_idxs = []

    for row, detection_idx in enumerate(
        detection_idxs
    ):
        corners = detections_by_index[
            detection_idx
        ]

        for marker_id, xy in corners.items():
            marker_id = int(
                marker_id
            )

            col = id_to_col[
                marker_id
            ]

            coords[
                row,
                col,
                :
            ] = np.asarray(
                xy,
                dtype=np.float32,
            )

        frame_idxs.append(
            int(
                frame_indices_by_index[
                    detection_idx
                ]
            )
        )

    payload = {
        "version": "2.0",
        "storage_method": "array",
        "marker_coords": [
            coords.tolist()
        ],
        "marker_ids": marker_ids,
        "detection_idxs": detection_idxs,
        "frame_idxs": [
            frame_idxs
        ],
    }

    return payload


def save_calibcam_payload(
    path: str | Path,
    payload: dict,
) -> Path:
    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        payload,
        allow_pickle=True,
    )

    return path


def load_calibcam_payload(
    path: str | Path,
) -> dict:
    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Detection payload not found: {path}"
        )

    return np.load(
        path,
        allow_pickle=True,
    ).item()


def summarize_calibcam_payload(
    payload: dict,
) -> dict:
    coords = np.asarray(
        payload["marker_coords"],
        dtype=np.float32,
    )

    return {
        "version": payload["version"],
        "storage_method": (
            payload["storage_method"]
        ),
        "marker_coords_shape": list(
            coords.shape
        ),
        "marker_ids": list(
            payload["marker_ids"]
        ),
        "n_marker_ids": len(
            payload["marker_ids"]
        ),
        "n_detection_idxs": len(
            payload["detection_idxs"]
        ),
        "detection_idx_min": int(
            min(
                payload["detection_idxs"]
            )
        ),
        "detection_idx_max": int(
            max(
                payload["detection_idxs"]
            )
        ),
        "frame_idx_min": int(
            min(
                payload["frame_idxs"][0]
            )
        ),
        "frame_idx_max": int(
            max(
                payload["frame_idxs"][0]
            )
        ),
    }
