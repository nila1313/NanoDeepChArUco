# Two-Stage Linux Portability Validation

## 1. Purpose

This document validates that the fixed-intrinsics two-stage calibration
workflow can be reproduced on a second operating system and machine.

The reference development and validation platform was macOS.

The portability platform was:

- host: bbo3008
- operating system: Linux
- repository branch: feature/two-stage-calibration
- validated commit: ed14644
- CalibCam version: 4.2.0

The goal was to verify that the same repository revision, calibration
inputs, board definition, synchronization, and CalibCam configuration
produce equivalent stereo geometry on Linux.

## 2. Repository Verification

The Linux checkout was switched from the previous detached v1.1.0 state
to:

    feature/two-stage-calibration

at commit:

    ed14644

The working tree was clean before validation.

The NanoDeepChArUco Python import resolved directly to the Linux checkout:

    /home/chaity/Documents/NanoDeepChArUco/src/nanodeepcharuco

The feature CLI exposed the expected options, including:

- calibration_single
- auto_sync
- fixed_sync
- fisheye_equidistant

## 3. Software Tests

The complete test suite was executed on Linux.

Result:

    37 passed

`git diff --check` completed without errors.

## 4. Stage-A Intrinsic Inputs

The exact validated Stage-A calibration files from the macOS validation
were transferred to Linux.

SHA-256:

Left:

    dbaf04beb91fdd6756726d0f2b7c015db983513e52c2d7951cada5cb75422ce9

Right:

    a137bf2fd1b1d5eabc1ce8ceaa6b597a3adff00afd9b39ff3910f83f7b607498

The hashes were identical on macOS and Linux.

## 5. Stage-B Dataset

Dataset:

    20230613/4pi/030_checkerboard_1

Videos:

- CADDX000013_left.MP4
- CADDX000013_right.MP4

Synchronization:

    Left(t) <-> Right(t+1)

Frame configuration:

- frames_start: 0
- frames_end: 2640
- frames_step: 20
- frames_offsets: 0 1

The resulting detection payload contained:

- 131 shared stereo frames
- logical frame range: 0 to 2600
- left physical range: 0 to 2600
- right physical range: 1 to 2601

## 6. Stage-B Calibration Configuration

The Linux run used the physical small-board asset:

    configs/boards/small_5x6_dict6x6_250_meters.npy

The calibration backend was official CalibCam 4.2.

The Stage-B command used:

- two Stage-A calibration files through calibration_single
- calibration_multi
- models: omnidir
- projection: fisheye_equidistant
- multi_vars: extrinsics extrinsics

Therefore Stage-B optimized stereo extrinsics while keeping the Stage-A
intrinsics fixed.

## 7. Linux Calibration Result

The CalibCam optimization converged successfully.

Results:

- detection rejection: 0.00%
- frame rejection: 0.00%
- final multicam calibration created successfully

Linux camera-1 extrinsics:

rvec_cam1:

    [-0.004009505719971617,
     -0.3355456048136775,
     -0.2640129491282213]

tvec_cam1:

    [-0.02809265727030375,
      0.0055389691175134715,
     -0.006983286781077278]

Baseline:

    29.472764811912146 mm

Rotation-vector magnitude:

    24.46400741198571 degrees

## 8. Fixed-Intrinsics Verification

The final Linux Stage-B calibration was compared directly with the two
Stage-A input calibrations.

For the left camera:

- A: exact equality
- k: exact equality
- xi: exact equality
- maximum absolute difference: 0

For the right camera:

- A: exact equality
- k: exact equality
- xi: exact equality
- maximum absolute difference: 0

This confirms that the Linux Stage-B run preserved the Stage-A
intrinsics exactly.

## 9. macOS vs Linux Comparison

The corresponding validated macOS fixed-offset result was:

Baseline:

    29.47282736159845 mm

Rotation-vector magnitude:

    24.463975125217516 degrees

Cross-platform differences:

- translation-vector difference:
  0.0000680720117555641 mm

- baseline difference:
  0.0000625496863020103 mm

- relative rotation difference:
  0.0000371955383787638 degrees

These differences are extremely small and are consistent with numerical
floating-point variation between platforms.

## 10. Validation Conclusion

The fixed-intrinsics two-stage calibration workflow was successfully
reproduced on Linux at commit ed14644.

The Linux run:

- used the same physical Stage-B board definition;
- used byte-identical Stage-A calibration inputs;
- reproduced the same 131 fixed-offset stereo pairs;
- used official CalibCam 4.2;
- used omnidir with fisheye_equidistant projection;
- optimized extrinsics only;
- preserved A, k, and xi exactly;
- converged successfully;
- produced stereo geometry essentially identical to the macOS result.

This validates cross-platform portability for the tested Pair 01
fixed-offset two-stage workflow.

This validation does not establish identical behavior for every
hardware, operating system, dataset, or synchronization configuration.
