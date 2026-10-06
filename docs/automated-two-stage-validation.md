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

## Multi-dataset validation

The automatic synchronization stage was subsequently tested on additional
small-board stereo recordings.

| Dataset | Automatically selected offset | Stage-2 result |
|---|---:|---|
| Pair 01 | +1 | 67 stable stereo pairs |
| Pair 03 | -1 | 60 stable stereo pairs |
| Pair 05 | -5 | 11 stable stereo pairs with adaptive stability |
| Pair 07 | +7 | 11 stable pairs before residual cleanup |

These tests exercise different temporal alignments rather than assuming one
fixed camera offset.

### Adaptive stable-board selection

A fixed 2 px motion threshold was sufficient for Pair 01 and Pair 03, but was
too restrictive for Pair 05 and Pair 07.

Adaptive stability uses the observed ChArUco motion distribution, with
`stable_motion_px` as the minimum threshold and `stable_motion_max_px` as the
upper limit.

### Pair 05

Pair 05 selected offset `-5` automatically.

Adaptive thresholds were approximately:

- left: `5.108 px`
- right: `5.997 px`

The stable-board filter retained 11 stereo pairs.

The resulting calibration converged with:

- median residual, camera 0: approximately `0.40 px`
- median residual, camera 1: approximately `0.62 px`
- maximum residual, camera 0: approximately `2.83 px`
- maximum residual, camera 1: approximately `3.13 px`

### Pair 07 automatic outlier cleanup

Pair 07 selected offset `+7` automatically.

Adaptive thresholds were approximately:

- left: `4.716 px`
- right: `5.307 px`

The stable-board filter retained 11 stereo pairs.

The initial Stage-2 calibration contained one severe residual outlier:

`1200 -> 1207`

The automatic residual analysis rejected only this stereo pair and reran
CalibCam with the remaining 10 pairs.

The cleaned calibration converged with:

- median residual, camera 0: `0.41 px`
- median residual, camera 1: `0.60 px`
- maximum residual, camera 0: `3.01 px`
- maximum residual, camera 1: `3.00 px`

The geometry changed only slightly after cleanup:

- baseline change: `+0.149 mm`
- rotation-magnitude change: `-0.107 deg`

## Config-based execution

The automated Stage-2 settings can be stored in:

`configs/pipelines/two_stage.yaml`

A typical invocation is:

```bash
python -m nanodeepcharuco \
  --config configs/pipelines/two_stage.yaml \
  --videos LEFT_VIDEO.mp4 RIGHT_VIDEO.mp4 \
  --stage1_intrinsics LEFT_INTRINSICS.yml RIGHT_INTRINSICS.yml \
  --frames_end FRAME_COUNT \
  --device cuda \
  --calibcam_python /path/to/calibcam/python \
  --calibcam_board /path/to/calibcam_compatible_board.npy \
  --data_path runs/my_calibration
```

Command-line arguments override values stored in the YAML profile.

## Conclusion

The automated Stage-2 workflow has now been validated across four real small-board stereo recordings with automatically recovered offsets of `+1`, `-1`, `-5`, and `+7`.

Across these tests the pipeline demonstrated automatic temporal-offset recovery, adaptive stable-board selection, fixed Stage-1 intrinsics, extrinsics-only optimization, residual-based stereo-pair rejection, and cleaned recalibration.

These results validate the implementation across the tested recordings, but do not imply universal performance for every camera system or recording condition.
