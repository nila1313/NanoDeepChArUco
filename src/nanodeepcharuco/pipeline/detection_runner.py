from __future__ import annotations

from pathlib import Path
import cv2

from nanodeepcharuco.calibcam.payload import build_calibcam_dict, save_calibcam_detection


def run_detector_on_video(video_path: str | Path, detector, frames_step: int,
                          side_name: str, frames_start: int = 0,
                          frames_end: int | None = None, frame_offset: int = 0):
    """Run a detector and key results by physical video frame index."""
    video_path = Path(video_path).expanduser().resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stop = frame_count if frames_end is None else frames_end
        first = max(0, frames_start)
        side_data = {}
        for aligned_frame_id in range(first, stop, frames_step):
            physical_frame_id = aligned_frame_id + frame_offset

            if physical_frame_id < 0 or physical_frame_id >= frame_count:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, physical_frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            if hasattr(detector, "process_frame"):
                corners = detector.process_frame(
                    frame, f"{side_name}_{physical_frame_id:08d}"
                ).corners
            else:
                corners = detector.detect_dict(frame)

            if not hasattr(detector, "process_frame") and corners and side_data:
                previous = side_data[next(reversed(side_data))]
                common_ids = sorted(set(previous) & set(corners))
                from nanodeepcharuco.detection.opencv_charuco import (
                    INTER_FRAME_DIST, MIN_CORNERS, check_detections_nondegenerate,
                )
                if check_detections_nondegenerate(
                    int(detector.params["boardWidth"]), common_ids, MIN_CORNERS
                ):
                    distances = [
                        float(((previous[idx] - corners[idx]) ** 2).sum() ** 0.5)
                        for idx in common_ids
                    ]
                    if distances and max(distances) < INTER_FRAME_DIST:
                        print(f"{side_name} frame {frame_id}: skipped (<3 px)")
                        continue
            side_data[int(physical_frame_id)] = corners
            print(
                f"{side_name} frame {physical_frame_id}: "
                f"{len(corners)} corners"
            )
        return side_data
    finally:
        cap.release()


def build_and_save_payload(
    side_data,
    output_path: str | Path,
    frame_offset: int = 0,
    frames_start: int = 0,
    frames_step: int = 1,
):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_calibcam_dict(
        side_data,
        frame_offset=frame_offset,
        frames_start=frames_start,
        frames_step=frames_step,
    )

    save_calibcam_detection(output_path, payload)
    return payload
