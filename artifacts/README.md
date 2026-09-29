# Generated Artifacts

Runtime outputs are kept here and excluded from version control.

- `runs/`: current training logs and launch metadata.
- `checkpoints/`: retained model checkpoints, including legacy experiments.
- `cache/training/`: decoded volume and mask cache.
- `exports/`: generated NIfTI files and benchmark inputs.

Create a new run directory rather than overwriting an existing checkpoint. Reports in `docs/` may link to local figures stored with a run.
