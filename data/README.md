# Local Data

This directory contains local research data and is excluded from version control.

- `training/joint/`: 1,302 paired susceptibility and mask tensors (651 original and 651 geometric) used by the maintained pipeline.
- `training/legacy/`: earlier generated training tensors.
- `sources/qsm/` and `sources/separate/`: source NIfTI volumes.
- `evaluation/`: MAT and NIfTI inputs used by scripts in `experiments/evaluation/`.
- `archives/`: original downloaded archives and extraction metadata.

Do not commit subject data. A new dataset needs a subject manifest unless it uses the documented original 651-volume joint mapping.
The geometric generator adds 651 paired volumes to `training/joint/` and writes
`training/joint/subjects.json` for the expanded dataset. Use that manifest with
`--source-manifest` when training from another directory; the default loader
discovers it beside the tensors.
