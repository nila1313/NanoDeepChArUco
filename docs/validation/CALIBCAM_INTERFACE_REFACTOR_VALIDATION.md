# CalibCam Interface Refactor Validation

## Status

PASS — the CalibCam-interface refactor preserves the validated
NanoDeepChArUco calibration workflow while exposing a cleaner
CalibCam-compatible data layout.

Validation completed: 2026-09-18.

Validated branch:

    refactor/calibcam-interface

Validated implementation state before this report:

    1bf2ef1 — Document CalibCam-compatible output layout

## Purpose

The refactor reorganized the application without intentionally changing
the Nano + DeepChArUco detection algorithm.

The main goals were:

- keep the command-line interface small and readable
- separate detection and calibration orchestration
- isolate the official CalibCam backend integration
- place CalibCam-compatible files directly at `data_path`
- place NanoDeepChArUco-specific metadata under
  `data_path/nanodeepcharuco`
- preserve synchronization, detection, and calibration behavior

## Automated tests

The refactored branch passes:

    29 / 29 unit tests

The suite covers:

- CLI defaults and synchronization-mode selection
- automatic synchronization orchestration
- fixed synchronization orchestration
- detector construction
- ChArUco board corner IDs
- detection pipeline orchestration
- calibration pipeline orchestration
- CalibCam command construction
- CalibCam run layout
- CalibCam detection payload construction
- YAML detection payload round-trip
- logical detection IDs versus physical frame IDs

## CalibCam detection compatibility

A synthetic detection payload was loaded successfully by the pinned
official CalibCamLib environment.

Verified backend:

    CalibCam version:    4.2.0
    CalibCam commit:     f51aa60961b6a9a7abea8b7c377dd5dfa7599f50
    CalibCamLib version: 0.5.2
    CalibCamLib commit:  5e3888b0647a4e20cf54ec54734ec6ed770a2169

Verified detection schema:

    version: 2.0
    storage_method: array
    marker_coords
    marker_ids
    detection_idxs
    frame_idxs

Logical detection indices and physical video-frame indices remain
separate, as required for synchronized stereo input.

## Output-layout validation

Official CalibCam inputs and outputs are written directly at `data_path`:

    detection_000.yml
    detection_001.yml
    calibration_single_000.yml
    calibration_single_001.yml
    joinedsingles_calibraton.yml
    joinedsingles_calibraton.npy
    joinedsingles_calibraton.mat
    multicam_calibration.yml
    multicam_calibration.npy
    multicam_calibration.mat
    multicam_calibration_board_positions.yml
    detections_cam_000.svg
    detections_cam_001.svg

NanoDeepChArUco-specific files are isolated under:

    nanodeepcharuco/
        resolved_config.yml
        input_manifest.yml
        tmp/

Automatic synchronization may additionally produce:

    nanodeepcharuco/sync_window_results.yml
    nanodeepcharuco/sync_segments.yml
    nanodeepcharuco/sync_pairs.yml

## Pair 01 fixed-offset regression

Dataset:

    20230613/4pi/030_checkerboard_1

Board:

    small_5x6
    DICT_6X6_250
    20 ChArUco corners

Alignment:

    Left(t) <-> Right(t+1)

Configuration:

    frames_start: 0
    frames_step: 20
    frames_offsets: [0, 1]
    models: [omnidir, omnidir]
    projection: perspective

The archived pre-refactor run and the refactored run used the same:

    131 synchronized frames
    logical frame range: 0 to 2600
    left physical range: 0 to 2600
    right physical range: 1 to 2601
    marker IDs
    board assets
    Nano executable
    DeepChArUco checkpoint
    RefineNet checkpoint
    DeepChArUco configuration

### Archived result

CalibCam median objective residuals:

    camera 0: 0.2035980224609375 px
    camera 1: 0.3076171875 px

Stereo baseline:

    0.029549237826622703

### Refactored canonical-luminance result

CalibCam median objective residuals:

    camera 0: 0.2100830078125 px
    camera 1: 0.318359375 px

Difference from archived result:

    camera 0: +0.0064849853515625 px
    camera 1: +0.0107421875 px

Stereo result remained very close to the archived solution.

### Legacy-BGR comparison

A second refactored detection pass was run with:

    --no_canonical_luma

Among coordinates present in both archived and legacy-BGR detections,
the median and 95th-percentile coordinate differences were 0 px for
both cameras.

The legacy-BGR final calibration produced:

    camera 0: 0.211029052734375 px
    camera 1: 0.3167572021484375 px

Difference from archived result:

    camera 0: +0.0074310302734375 px
    camera 1: +0.0091400146484375 px

This shows that canonical luminance changes some borderline detections,
but it is not responsible for a meaningful degradation in final
calibration quality.

The exact historical video path used by the archived fixed-offset run is
no longer present, so bit-for-bit reproduction of that historical run
cannot be established.

## Full-frame Pair 01 regression

A full-frame fixed-offset run also completed successfully.

Shared stereo detections:

    2617

Final board poses after CalibCam filtering:

    2615

CalibCam median objective residuals:

    camera 0: 0.204681396484375 px
    camera 1: 0.31890869140625 px

Final stereo pose:

    rvec:
    [0.010360962075730272,
     -0.32744916892318815,
     -0.2637631784453921]

    tvec:
    [-0.028435559923264807,
      0.005471198604847612,
     -0.00604029298894674]

    baseline:
    0.029580402663175315

CalibCam completed all optimization stages successfully.

## Conclusion

The refactor preserves the functional behavior of the pipeline.

Validated properties include:

- synchronization semantics preserved
- physical/logical frame mapping preserved
- detection payload format preserved
- official CalibCam 4.2 compatibility preserved
- official CalibCam output naming preserved
- calibrated stereo geometry preserved
- final reprojection quality preserved
- NanoDeepChArUco metadata cleanly separated from CalibCam files
- generated run data remains excluded from Git
- 29 automated tests pass

The validation supports treating the refactor as a structural and
interface improvement rather than an algorithm change.

The implementation is not claimed to reproduce every historical run
bit-for-bit across different decoder, hardware, or runtime environments.
