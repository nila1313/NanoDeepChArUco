from __future__ import annotations

import cv2
import numpy as np


DEFAULT_GAMMA_DARK = 1.4


def gamma_dark(
    frame: np.ndarray,
    gamma: float = DEFAULT_GAMMA_DARK,
) -> np.ndarray:
    """
    Apply the verified gamma-dark transform used by
    the NanoDeepChArUco hybrid pipeline.
    """

    if frame is None:
        raise ValueError(
            "Input frame is None"
        )

    if gamma <= 0:
        raise ValueError(
            "Gamma must be greater than zero"
        )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    x = (
        gray.astype(np.float32)
        / 255.0
    )

    y = np.power(
        x,
        float(gamma),
    )

    y = np.clip(
        y * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        y,
        cv2.COLOR_GRAY2BGR,
    )
