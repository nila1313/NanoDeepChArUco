from __future__ import annotations

from pathlib import Path

import yaml


def save_sync_reports(
    run_dir,
    window_results,
    segments,
    synchronized_pairs,
):
    run_dir = Path(run_dir)

    window_payload = []

    for result in window_results:
        item = {
            "start_frame": int(
                result.start_frame
            ),
            "end_frame_exclusive": int(
                result.end_frame
            ),
            "best_offset": (
                int(result.best_offset)
                if result.best_offset is not None
                else None
            ),
            "inlier_ratio_gap": float(
                result.inlier_ratio_gap
            ),
            "n_left_frames": int(
                result.n_left_frames
            ),
            "n_right_frames": int(
                result.n_right_frames
            ),
        }

        if result.best_score is not None:
            item["best_score"] = {
                "offset": int(
                    result.best_score.offset
                ),
                "n_frame_pairs": int(
                    result.best_score.n_frame_pairs
                ),
                "n_correspondences": int(
                    result.best_score.n_correspondences
                ),
                "n_inliers": int(
                    result.best_score.n_inliers
                ),
                "inlier_ratio": float(
                    result.best_score.inlier_ratio
                ),
                "median_sampson_error": float(
                    result.best_score.median_sampson_error
                ),
                "mean_sampson_error": float(
                    result.best_score.mean_sampson_error
                ),
            }
        else:
            item["best_score"] = None

        if result.second_score is not None:
            item["second_score"] = {
                "offset": int(
                    result.second_score.offset
                ),
                "n_frame_pairs": int(
                    result.second_score.n_frame_pairs
                ),
                "n_correspondences": int(
                    result.second_score.n_correspondences
                ),
                "n_inliers": int(
                    result.second_score.n_inliers
                ),
                "inlier_ratio": float(
                    result.second_score.inlier_ratio
                ),
                "median_sampson_error": float(
                    result.second_score.median_sampson_error
                ),
                "mean_sampson_error": float(
                    result.second_score.mean_sampson_error
                ),
            }
        else:
            item["second_score"] = None

        window_payload.append(item)

    segment_payload = [
        {
            "start_frame": int(
                segment.start_frame
            ),
            "end_frame_exclusive": int(
                segment.end_frame
            ),
            "offset": int(
                segment.offset
            ),
            "n_support_windows": int(
                segment.n_support_windows
            ),
            "mean_inlier_ratio": float(
                segment.mean_inlier_ratio
            ),
            "mean_ratio_gap": float(
                segment.mean_ratio_gap
            ),
        }
        for segment in segments
    ]

    pair_payload = [
        {
            "detection_idx": int(
                pair.left_frame
            ),
            "left_frame": int(
                pair.left_frame
            ),
            "right_frame": int(
                pair.right_frame
            ),
            "offset": int(
                pair.offset
            ),
            "segment_start": int(
                pair.segment_start
            ),
            "segment_end_exclusive": int(
                pair.segment_end
            ),
        }
        for pair in synchronized_pairs
    ]

    window_path = (
        run_dir
        / "sync_window_results.yml"
    )

    segment_path = (
        run_dir
        / "sync_segments.yml"
    )

    pair_path = (
        run_dir
        / "sync_pairs.yml"
    )

    with open(window_path, "w") as f:
        yaml.safe_dump(
            window_payload,
            f,
            sort_keys=False,
        )

    with open(segment_path, "w") as f:
        yaml.safe_dump(
            segment_payload,
            f,
            sort_keys=False,
        )

    with open(pair_path, "w") as f:
        yaml.safe_dump(
            pair_payload,
            f,
            sort_keys=False,
        )

    return (
        window_path,
        segment_path,
        pair_path,
    )
