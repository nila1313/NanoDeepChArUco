from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml


def analyze_board_positions(
    path: Path,
    *,
    mad_multiplier: float = 6.0,
    min_threshold_px: float = 6.0,
    max_threshold_px: float = 10.0,
) -> dict:
    """
    Analyze CalibCam's final Stage-2 residuals per stereo pair.

    The residual layout in
    multicam_calibration_board_positions.yml is:

        camera x frame x board_point x (x, y)

    Pair score = the larger maximum corner residual from the
    two cameras for that synchronized stereo pair.
    """

    path = Path(path)

    with path.open() as f:
        data = yaml.safe_load(f)

    frame_idxs = np.asarray(
        data["frame_idxs"],
        dtype=int,
    )

    if frame_idxs.ndim != 2:
        raise ValueError(
            "Expected frame_idxs with shape "
            "(n_cameras, n_frames)"
        )

    n_cams, n_frames = frame_idxs.shape

    if n_cams != 2:
        raise ValueError(
            "Stage-2 outlier rejection currently requires "
            "exactly two cameras"
        )

    fun = np.asarray(
        data["info"]["fun_final"],
        dtype=float,
    )

    denominator = n_cams * n_frames * 2

    if fun.size % denominator != 0:
        raise ValueError(
            "Unexpected CalibCam residual layout"
        )

    n_points = fun.size // denominator

    residuals = fun.reshape(
        n_cams,
        n_frames,
        n_points,
        2,
    )

    # Match CalibCam's own reporting logic.
    residuals = np.abs(residuals)
    residuals[residuals == 0] = np.nan

    distances = np.linalg.norm(
        residuals,
        axis=-1,
    )

    max_error = np.full(
        (n_cams, n_frames),
        np.nan,
        dtype=float,
    )

    mean_error = np.full(
        (n_cams, n_frames),
        np.nan,
        dtype=float,
    )

    for cam_idx in range(n_cams):
        for frame_idx in range(n_frames):
            values = distances[
                cam_idx,
                frame_idx,
            ]

            finite = values[
                np.isfinite(values)
            ]

            if finite.size == 0:
                continue

            max_error[
                cam_idx,
                frame_idx,
            ] = float(np.max(finite))

            mean_error[
                cam_idx,
                frame_idx,
            ] = float(np.mean(finite))

    pair_scores = []

    for idx in range(n_frames):
        values = max_error[:, idx]
        finite = values[np.isfinite(values)]

        if finite.size == 0:
            pair_scores.append(np.nan)
        else:
            pair_scores.append(
                float(np.max(finite))
            )

    pair_scores = np.asarray(
        pair_scores,
        dtype=float,
    )

    finite_scores = pair_scores[
        np.isfinite(pair_scores)
    ]

    if finite_scores.size == 0:
        raise ValueError(
            "No finite calibration residuals found"
        )

    median = float(
        np.median(finite_scores)
    )

    mad = float(
        np.median(
            np.abs(
                finite_scores - median
            )
        )
    )

    robust_threshold = (
        median
        + mad_multiplier * mad
    )

    threshold = float(
        min(
            max(
                robust_threshold,
                min_threshold_px,
            ),
            max_threshold_px,
        )
    )

    rejected_indices = [
        int(idx)
        for idx, score in enumerate(
            pair_scores
        )
        if (
            np.isfinite(score)
            and score > threshold
        )
    ]

    pairs = []

    for idx in range(n_frames):
        pairs.append(
            {
                "index": int(idx),
                "left_frame": int(
                    frame_idxs[0, idx]
                ),
                "right_frame": int(
                    frame_idxs[1, idx]
                ),
                "left_max_px": (
                    None
                    if not np.isfinite(
                        max_error[0, idx]
                    )
                    else float(
                        max_error[0, idx]
                    )
                ),
                "right_max_px": (
                    None
                    if not np.isfinite(
                        max_error[1, idx]
                    )
                    else float(
                        max_error[1, idx]
                    )
                ),
                "left_mean_px": (
                    None
                    if not np.isfinite(
                        mean_error[0, idx]
                    )
                    else float(
                        mean_error[0, idx]
                    )
                ),
                "right_mean_px": (
                    None
                    if not np.isfinite(
                        mean_error[1, idx]
                    )
                    else float(
                        mean_error[1, idx]
                    )
                ),
                "pair_score_px": (
                    None
                    if not np.isfinite(
                        pair_scores[idx]
                    )
                    else float(
                        pair_scores[idx]
                    )
                ),
                "rejected": (
                    idx in rejected_indices
                ),
            }
        )

    return {
        "source": str(path),
        "n_pairs": int(n_frames),
        "n_board_points": int(n_points),
        "mad_multiplier": float(
            mad_multiplier
        ),
        "min_threshold_px": float(
            min_threshold_px
        ),
        "max_threshold_px": float(
            max_threshold_px
        ),
        "median_pair_max_px": median,
        "mad_pair_max_px": mad,
        "robust_threshold_px": float(
            robust_threshold
        ),
        "used_threshold_px": threshold,
        "rejected_indices": (
            rejected_indices
        ),
        "rejected_count": len(
            rejected_indices
        ),
        "pairs": pairs,
    }


def _subset_detection_payload(
    payload: dict,
    keep_indices: list[int],
) -> dict:
    """
    Subset all frame-dependent fields while preserving global
    marker_ids and metadata.
    """

    out = dict(payload)

    n_frames = len(
        payload["detection_idxs"]
    )

    if len(
        payload["frame_idxs"][0]
    ) != n_frames:
        raise ValueError(
            "frame_idxs and detection_idxs "
            "length mismatch"
        )

    if len(
        payload["marker_coords"][0]
    ) != n_frames:
        raise ValueError(
            "marker_coords and detection_idxs "
            "length mismatch"
        )

    out["marker_coords"] = [[
        payload["marker_coords"][0][idx]
        for idx in keep_indices
    ]]

    out["detection_idxs"] = [
        payload["detection_idxs"][idx]
        for idx in keep_indices
    ]

    out["frame_idxs"] = [[
        payload["frame_idxs"][0][idx]
        for idx in keep_indices
    ]]

    return out


def write_cleaned_detection_payloads(
    detection_paths: tuple[Path, Path],
    output_dir: Path,
    rejected_indices: list[int],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payloads = [
        np.load(
            path,
            allow_pickle=True,
        )[()]
        for path in detection_paths
    ]

    n_frames = len(
        payloads[0]["detection_idxs"]
    )

    if len(
        payloads[1]["detection_idxs"]
    ) != n_frames:
        raise ValueError(
            "Left/right detection payload lengths differ"
        )

    rejected = set(
        int(idx)
        for idx in rejected_indices
    )

    keep_indices = [
        idx
        for idx in range(n_frames)
        if idx not in rejected
    ]

    output_paths = (
        output_dir
        / "detection_000.npy",
        output_dir
        / "detection_001.npy",
    )

    for payload, output_path in zip(
        payloads,
        output_paths,
    ):
        cleaned = _subset_detection_payload(
            payload,
            keep_indices,
        )

        np.save(
            output_path,
            cleaned,
            allow_pickle=True,
        )

    return output_paths


def compare_extrinsics(
    original_path: Path,
    cleaned_path: Path,
) -> dict:
    def load_cam1(path):
        with Path(path).open() as f:
            data = yaml.safe_load(f)

        cam1 = data["calibs"][1]

        rvec = np.asarray(
            cam1["rvec_cam"],
            dtype=float,
        )

        tvec = np.asarray(
            cam1["tvec_cam"],
            dtype=float,
        )

        return rvec, tvec

    old_rvec, old_tvec = load_cam1(
        original_path
    )

    new_rvec, new_tvec = load_cam1(
        cleaned_path
    )

    old_baseline = float(
        np.linalg.norm(old_tvec)
        * 1000.0
    )

    new_baseline = float(
        np.linalg.norm(new_tvec)
        * 1000.0
    )

    old_rotation = float(
        math.degrees(
            np.linalg.norm(old_rvec)
        )
    )

    new_rotation = float(
        math.degrees(
            np.linalg.norm(new_rvec)
        )
    )

    return {
        "original_baseline_mm": (
            old_baseline
        ),
        "cleaned_baseline_mm": (
            new_baseline
        ),
        "baseline_change_mm": (
            new_baseline - old_baseline
        ),
        "original_rotation_deg": (
            old_rotation
        ),
        "cleaned_rotation_deg": (
            new_rotation
        ),
        "rotation_change_deg": (
            new_rotation - old_rotation
        ),
    }


def save_report(
    report: dict,
    path: Path,
) -> None:
    Path(path).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
