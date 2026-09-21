# Two-Stage Automatic Synchronization Validation

## 1. Purpose

This document validates the automatic synchronization path of the
NanoDeepChArUco two-stage calibration workflow.

The goal is to verify that:

- automatic synchronization can recover the correct temporal alignment;
- the recovered synchronization is propagated correctly into the
  CalibCam-compatible detection payloads;
- Stage-B calibration keeps Stage-A intrinsics fixed;
- official CalibCam 4.2 converges using the automatically synchronized
  detections;
- the recovered stereo geometry is consistent across independent
  recordings.

The calibration backend remains official CalibCam 4.2.

## 2. Common Stage-B Configuration

The following configuration was used for both validation recordings:

- board: small 5x6 ChArUco
- dictionary: DICT_6X6_250
- physical square size: 0.02 m
- camera model: omnidir
- projection: fisheye_equidistant
- Stage-A intrinsics supplied through calibration_single
- multi-camera optimization enabled
- multi_vars: extrinsics extrinsics
- frame sampling step: 20
- automatic synchronization candidate offsets: -3 through +3

Stage-A camera intrinsics were taken from:

- left_multicam_calibration.yml
- right_multicam_calibration.yml

Stage-B was therefore an extrinsics-only optimization.

## 3. Pair 01

Dataset:

20230613/4pi/030_checkerboard_1

Videos:

- CADDX000013_left.MP4
- CADDX000013_right.MP4

### 3.1 Automatic Synchronization

Automatic synchronization selected one stable segment:

| Range | Offset | Support | Mean ratio | Mean gap |
| --- | ---: | ---: | ---: | ---: |
| 500-2699 | +1 | 13 | 0.7044 | 0.2252 |

The resulting physical relationship was:

Left(t) <-> Right(t+1)

The final synchronized detection payload contained:

- 61 trusted stereo pairs
- left physical range: 500 to 2600
- right physical range: 501 to 2601
- identical logical detection IDs between cameras
- unique physical R-L offset: +1

### 3.2 Stage-B Calibration

CalibCam completed successfully.

Results:

- exit status: 0
- optimization converged: yes
- detection rejection: 0.00%
- frame rejection: 0.00%
- baseline: 29.331716 mm
- camera-1 rotation-vector magnitude: 24.535119 degrees

Camera-1 extrinsics:

rvec_cam1:

    [-0.0045233654, -0.3372351435, -0.2638620301]

tvec_cam1:

    [-0.0279685358, 0.0054657769, -0.0069452019]

### 3.3 Frozen-Intrinsics Verification

For both cameras:

- A: exact equality with Stage-A calibration
- k: exact equality with Stage-A calibration
- xi: exact equality with Stage-A calibration
- maximum absolute difference: 0

### 3.4 Comparison with Fixed +1 Validation

The previously validated fixed synchronization used:

Left(t) <-> Right(t+1)

Fixed-sync result:

- 131 stereo pairs
- baseline: 29.472827 mm
- rotation-vector magnitude: 24.463975 degrees

Automatic-sync result:

- 61 trusted stereo pairs
- baseline: 29.331716 mm
- rotation-vector magnitude: 24.535119 degrees

Geometry difference:

- translation-vector difference: 0.149101 mm
- baseline difference: 0.141112 mm
- true relative rotation difference: 0.101194 degrees

Automatic synchronization therefore recovered the same temporal
relationship and produced closely agreeing stereo geometry while using
only trusted synchronization regions.

## 4. Pair 03

Dataset:

20230613/4pi/070_checkerboard_3

Videos:

- CADDX000022_left.MP4
- CADDX000022_right.MP4

The recording was verified to contain the small DICT_6X6_250 board.

### 4.1 Automatic Synchronization

Automatic synchronization produced three trusted segments:

| Range | Offset | Support | Mean ratio | Mean gap |
| --- | ---: | ---: | ---: | ---: |
| 100-799 | -1 | 5 | 0.8630 | 0.2595 |
| 900-1099 | -1 | 2 | 0.8576 | 0.2474 |
| 1200-2199 | -1 | 8 | 0.7546 | 0.3073 |

All trusted segments independently selected the same temporal
relationship:

Left(t) <-> Right(t-1)

The final synchronized detection payload contained:

- 71 trusted stereo pairs
- left physical range: 100 to 2140
- right physical range: 99 to 2139
- identical logical detection IDs between cameras
- unique physical R-L offset: -1

### 4.2 Stage-B Calibration

CalibCam completed successfully.

Results:

- exit status: 0
- optimization converged: yes
- detection rejection: 0.25%
- frame rejection: 4.23%
- baseline: 29.408466 mm
- camera-1 rotation-vector magnitude: 24.367379 degrees

Camera-1 extrinsics:

rvec_cam1:

    [-0.0131460940, -0.3340424827, -0.2628977647]

tvec_cam1:

    [-0.0280499399, 0.0054977783, -0.0069161535]

### 4.3 Frozen-Intrinsics Verification

For both cameras:

- A: exact equality with Stage-A calibration
- k: exact equality with Stage-A calibration
- xi: exact equality with Stage-A calibration
- maximum absolute difference: 0

## 5. Cross-Recording Comparison

The two automatic-synchronization experiments used independent stereo
recordings and recovered opposite one-frame temporal offsets.

| Metric | Pair 01 | Pair 03 |
| --- | ---: | ---: |
| Recovered offset | +1 | -1 |
| Trusted stereo pairs | 61 | 71 |
| Baseline | 29.331716 mm | 29.408466 mm |
| Rotation-vector magnitude | 24.535119 deg | 24.367379 deg |
| Intrinsics preserved exactly | yes | yes |
| Calibration converged | yes | yes |

Cross-recording geometry difference:

- translation-vector difference: 0.092166 mm
- baseline difference: 0.076750 mm
- true relative rotation difference: 0.526106 degrees

## 6. Validation Conclusion

The automatic synchronization path has been validated on two independent
small-board stereo recordings.

Pair 01 recovered:

Left(t) <-> Right(t+1)

Pair 03 recovered:

Left(t) <-> Right(t-1)

For both recordings:

- automatic synchronization produced trusted stereo pairs;
- the physical frame mapping matched the recovered offset;
- the synchronized detections were accepted by official CalibCam 4.2;
- Stage-B optimization converged;
- Stage-A A, k, and xi remained exactly fixed;
- the resulting stereo geometry was consistent across recordings.

These results validate the automatic-synchronization and fixed-intrinsics
two-stage workflow for the tested Pair 01 and Pair 03 recordings.

They do not establish correctness for every possible dataset or temporal
offset.
