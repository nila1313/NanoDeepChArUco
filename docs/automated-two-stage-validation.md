# Automated Two-Stage Calibration Validation

## Purpose

This branch extends the validated two-stage wide-angle calibration workflow with
automatic stereo synchronization and automatic stable-board frame selection for
Stage 2.

The calibration architecture remains unchanged:

1. Stage 1 calibrates each camera independently using the large ChArUco board.
2. Stage 2 uses the small ChArUco board for stereo calibration.
3. Stage-1 intrinsics are loaded into Stage 2 and remain fixed.
4. Only stereo extrinsics are optimized in Stage 2.

## Stage-1 inputs

Large-board dataset:

- Dataset: `20230613/4pi/040_checkerboard_2`
- Left Stage-1 calibration:
  `runs/two_stage/stage1_intrinsics/left_full_040_checkerboard_2/multicam_calibration.yml`
- Right Stage-1 calibration:
  `runs/two_stage/stage1_intrinsics/right_full_040_checkerboard_2/multicam_calibration.yml`

## Stage-2 validation dataset

Small-board dataset:

- Dataset: `20230613/4pi/030_checkerboard_1`
- Left video: `CADDX000013_left.MP4`
- Right video: `CADDX000013_right.MP4`
- Board: 5x6, DICT_6X6_250
- Real square size: 0.02 m
- Real marker size: 0.0133333333 m
- Frame sampling step: 20

## Automatic synchronization

Candidate offsets:

`[-3, -2, -1, 0, 1, 2, 3]`

The automatic synchronization stage identified the persistent trusted segment:

- frame range: 400–2599
- offset: `+1`

Therefore the selected stereo relationship was:

`Left(t) <-> Right(t+1)`

This agrees with the previously validated Pair-01 alignment.

## Stable-board selection

The synchronization stage produced 109 candidate stereo pairs.

After requiring low ChArUco-board motion in both cameras:

- candidate synchronized pairs: 109
- accepted stable stereo pairs: 67

All final selected physical frame pairs satisfy:

`right_frame - left_frame = +1`

## Fixed-intrinsic Stage 2

Stage 2 was run with CalibCam 4.2.0 using:

- `--calibration_single LEFT_STAGE1 RIGHT_STAGE1`
- `--calibration_multi`
- `--multi_vars extrinsics extrinsics`

The final calibration confirmed that Stage-1 intrinsics remained exactly fixed:

### Camera 0

- max absolute A difference: `0.0`
- max absolute k difference: `0.0`
- max absolute xi difference: `0.0`

### Camera 1

- max absolute A difference: `0.0`
- max absolute k difference: `0.0`
- max absolute xi difference: `0.0`

## Final stereo result

Camera-1 rotation vector:

`[-0.0044951454, -0.3354975529, -0.2646100218]`

Camera-1 translation vector in meters:

`[-0.0280825927, 0.0055125759, -0.0069976819]`

Derived values:

- baseline: `29.4616 mm`
- rotation magnitude: `24.4833 deg`

Calibration quality:

- median reprojection residual, camera 0: approximately `0.30 px`
- median reprojection residual, camera 1: approximately `0.41 px`
- maximum residual, camera 0: approximately `1.63 px`
- maximum residual, camera 1: approximately `4.12 px`
- discarded detections: `0.00%`
- discarded frames: `0.00%`
- optimization converged: `True`
- final cost: approximately `6.1805e+02`

## Conclusion

The automated Stage-2 workflow successfully:

- detected the expected Pair-01 temporal offset automatically,
- rejected ambiguous synchronization windows,
- selected stable small-board observations,
- generated correctly paired CalibCam detection payloads,
- preserved Stage-1 intrinsics exactly,
- and recovered stereo extrinsics consistent with the previously validated
  Pair-01 geometry.

This validates the automated two-stage workflow on Pair 01.

The current validation is dataset-specific and does not establish universal
performance across all camera pairs or recordings.
