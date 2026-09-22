from __future__ import annotations

import numpy as np


def build_calibcam_dict(
    side_data: dict[int, dict[int, np.ndarray]],
    frame_offset: int = 0,
    frames_start: int = 0,
    frames_step: int = 1,
) -> dict:
    """
    Build a CalibCam-compatible detection payload.

    side_data keys are PHYSICAL video-frame indices.

    CalibCam convention
    -------------------
    detection_idxs
        Index in the common sampled stereo sequence.

        Example with frames_start=0, frames_step=20:

            aligned frame 0  -> detection index 0
            aligned frame 20 -> detection index 1
            aligned frame 40 -> detection index 2

        These indices must match between cameras for a stereo
        observation.

    frame_idxs
        Actual physical video frame used by the camera.

        With right-camera offset +1:

            left:
                detection_idx 0 -> physical frame 0

            right:
                detection_idx 0 -> physical frame 1

    Relationship
    ------------
        physical_frame = aligned_frame + frame_offset
        aligned_frame  = physical_frame - frame_offset
    """
    if frames_step <= 0:
        raise ValueError("frames_step must be greater than zero")

    physical_frame_ids = sorted(side_data)

    marker_ids = sorted({
        int(marker_id)
        for frame_id in physical_frame_ids
        for marker_id in side_data[frame_id]
    })

    id_to_col = {
        marker_id: idx
        for idx, marker_id in enumerate(marker_ids)
    }

    coords = np.full(
        (
            len(physical_frame_ids),
            len(marker_ids),
            2,
        ),
        np.nan,
        dtype=np.float32,
    )

    detection_idxs = []
    frame_idxs = []

    for row, physical_frame in enumerate(physical_frame_ids):
        physical_frame = int(physical_frame)

        aligned_frame = (
            physical_frame
            - int(frame_offset)
        )

        delta = (
            aligned_frame
            - int(frames_start)
        )

        if delta < 0:
            raise ValueError(
                "Physical frame maps before frames_start: "
                f"physical={physical_frame}, "
                f"offset={frame_offset}, "
                f"aligned={aligned_frame}, "
                f"frames_start={frames_start}"
            )

        if delta % int(frames_step) != 0:
            raise ValueError(
                "Frame does not lie on the requested sampling grid: "
                f"physical={physical_frame}, "
                f"aligned={aligned_frame}, "
                f"frames_start={frames_start}, "
                f"frames_step={frames_step}"
            )

        detection_idx = (
            delta // int(frames_step)
        )

        detection_idxs.append(
            int(detection_idx)
        )

        # CalibCam frame_idxs preserve the physical video frame.
        frame_idxs.append(
            physical_frame
        )

        for marker_id, xy in side_data[physical_frame].items():
            col = id_to_col[int(marker_id)]

            coords[row, col, :] = np.asarray(
                xy,
                dtype=np.float32,
            )

    return {
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


def save_calibcam_detection(
    path,
    payload: dict,
):
    """Save a CalibCam-compatible NumPy detection file."""
    np.save(
        path,
        payload,
        allow_pickle=True,
    )


def load_calibcam_detection(
    path,
) -> dict:
    """Load a CalibCam-compatible NumPy detection file."""
    return np.load(
        path,
        allow_pickle=True,
    ).item()
