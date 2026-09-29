# Local Data

This directory contains local research data and is excluded from version control.

- `training/joint/`: 651 paired susceptibility and mask tensors used by the maintained pipeline.
- `training/legacy/`: earlier generated training tensors.
- `sources/qsm/` and `sources/separate/`: source NIfTI volumes.
- `evaluation/`: MAT and NIfTI inputs used by scripts in `experiments/evaluation/`.
- `archives/`: original downloaded archives and extraction metadata.

Do not commit subject data. A new dataset needs an explicit subject manifest unless it uses the documented 651-volume joint mapping.
