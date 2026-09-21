# Calibration board assets

## Small 5x6 physical calibration board

Repository asset:

`small_5x6_dict6x6_250_meters.npy`

This file is the NumPy-compatible repository representation of the
original calibration asset:

`board_small_exact_meters.npy`

Original reference SHA-256:

`69ef6a2f5708b978483b278d43b374a3576f908b5adfb71da8afdfab8e684ff5`

The repository representation was verified to contain the same
calibration semantics:

- board width: 5
- board height: 6
- ArUco dictionary: DICT_6X6_250 (`dictionary_type = 10`)
- normalized square size: 1.0
- normalized marker size: 0.6666666666666666
- physical square size: 0.02 m
- physical marker size: 0.013333333333333332 m
- physical square size X: 0.019926 m
- physical square size Y: 0.02 m
- unit: meters
- legacy: false

The byte-level SHA-256 of the repository `.npy` may differ from the
original reference because NumPy/pickle serialization is not used as
the semantic identity check. The physical board parameters above are
the calibration contract.

This physical-meter board must be used for the Stage-2 stereo
extrinsics calibration so that the recovered translation has the
correct physical scale.
