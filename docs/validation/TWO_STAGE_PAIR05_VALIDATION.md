# Two-Stage Pair 05 Validation

## Purpose

This document records an additional validation of the two-stage
fixed-intrinsics calibration workflow using a third independent stereo
recording.

## Dataset

Recording:

    20230615/4pi/040_checkerboard_1

Videos:

- CADDX000010_left.MP4
- CADDX000010_right.MP4

Video properties:

- resolution: 1920x1080
- frame rate: 60 FPS
- frames: 3720 per camera
- duration: 62 seconds

## Board Verification

Sample frames were tested against:

- DICT_6X6_250
- DICT_4X4_50

The recording consistently produced detections for DICT_6X6_250 and
therefore matches the small 5x6 Stage-B board.

The calibration used:

    configs/boards/small_5x6_dict6x6_250_meters.npy

## Automatic Synchronization

The synchronization search was widened to:

    -10 ... +10 frames

Stable synchronization segments:

    260-859:
        offset = -5
        support = 5
        mean_ratio = 0.6327
        mean_gap = 0.1472

    1660-1859:
        offset = -5
        support = 2
        mean_ratio = 0.7351
        mean_gap = 0.2004

The selected alignment was:

    Left(t) <-> Right(t-5)

Final synchronization result:

- requested calibration frames: 186
- available calibration frames: 127
- synchronized stereo pairs: 35
- synchronized logical range: 260 to 1840
- left physical range: 260 to 1840
- right physical range: 255 to 1835

Only trusted synchronization windows were used.

## Stage-B Calibration Configuration

The Stage-B calibration used:

- previously validated Stage-A left intrinsic calibration
- previously validated Stage-A right intrinsic calibration
- official CalibCam 4.2
- model: omnidir
- projection: fisheye_equidistant
- multi-camera calibration enabled
- optimization variables: extrinsics only

Equivalent CalibCam semantics:

    --calibration_single LEFT.yml RIGHT.yml
    --calibration_multi
    --models omnidir
    --projection fisheye_equidistant
    --multi_vars extrinsics extrinsics

## Calibration Quality

Pose estimation:

- camera 0: 34 frames
- camera 1: 34 frames
- initialization poses selected: 6

Optimization:

- initial cost: 3.9106e+06
- final cost: 2.1760e+03
- optimization converged: True
- discarded detections: 0.00%
- discarded frames: 0.00%

## Pair 05 Extrinsics

Camera-1 rotation vector:

    [-0.013093753149960786,
     -0.3244641520131202,
     -0.2665881348345257]

Camera-1 translation vector:

    [-0.028337660623371667,
      0.005742626829526552,
     -0.007077821951186988]

Baseline:

    29.76737032526205 mm

Rotation-vector magnitude:

    24.072251964914916 degrees

## Fixed-Intrinsics Verification

The final Stage-B result was compared with the Stage-A intrinsic inputs.

Left camera:

- A: exact equality
- k: exact equality
- xi: exact equality
- maximum absolute difference: 0

Right camera:

- A: exact equality
- k: exact equality
- xi: exact equality
- maximum absolute difference: 0

This confirms that Stage-B optimized only the stereo extrinsics and did
not modify the Stage-A intrinsic parameters.

## Relation to Previous Validation Recordings

Automatic synchronization results across the validated recordings:

    Pair 01: offset +1
    Pair 03: offset -1
    Pair 05: offset -5

Pair 05 has fewer trusted synchronized pairs than Pair 01 and Pair 03,
so it provides weaker geometric coverage. However, the complete
two-stage workflow still converged successfully and preserved the
Stage-A intrinsics exactly.

## Conclusion

Pair 05 provides an additional successful validation of the automatic
synchronization and fixed-intrinsics two-stage calibration workflow.

For this recording:

- the small Stage-B board was confirmed;
- a wide synchronization search selected offset -5;
- two independent trusted segments supported the same offset;
- 35 synchronized stereo pairs were retained;
- official CalibCam 4.2 converged successfully;
- no detections or frames were rejected;
- Stage-A A, k, and xi remained exactly unchanged;
- valid stereo extrinsics were produced.

This result extends the tested synchronization cases to +1, -1, and -5
frame offsets across three independent recordings.
