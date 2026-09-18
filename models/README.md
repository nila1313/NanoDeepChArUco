# Model Assets

Model checkpoints are tracked using Git LFS rather than ordinary Git blobs.

Expected layout:

```text
models/
├── deepcharuco/
│   ├── large_7x7/
│   │   └── detector.ckpt
│   └── small_5x6/
│       └── detector.ckpt
└── refinenet/
    └── refinenet.ckpt
```

Profiles in `configs/profiles/` reference these paths.

Verified model compatibility:

- `large_7x7`
  - board: DICT_4X4_50
  - ChArUco IDs: 36
  - detector output head: 37 channels

- `small_5x6`
  - board: DICT_6X6_250
  - ChArUco IDs: 20
  - detector output head: 21 channels

The RefineNet checkpoint is shared by both profiles.

After cloning, retrieve the checkpoint contents with:

```bash
git lfs pull
```

The repository `setup.sh` script performs this step automatically.
