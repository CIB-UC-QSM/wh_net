# Repository Guidelines

## Project Structure & Module Organization

Maintained Python code uses a `src` layout:

- `src/whnet/model.py`: proximal 3D U-Nets and legacy/optimized ADMM models.
- `src/whnet/data.py`: volume loading, caching, augmentation, and synthesis.
- `src/whnet/physics.py`: dipole kernels, metrics, and visualization helpers.
- `src/whnet/training.py`: joint training, validation, scheduling, and checkpoints.
- `tests/`: automated regression tests.
- `scripts/`: data preparation.
- `experiments/`: exploratory and legacy workflows.
- `docs/`: training documentation and reports.

Local datasets belong in `data/`; generated runs and checkpoints belong in `artifacts/`.

## Build, Test, and Development Commands

Run commands from the repository root:

- `uv sync`: install the locked Python 3.12 environment and local package.
- `uv run python -m unittest discover -s tests -v`: run regression tests.
- `uv run ruff check src tests scripts`: lint maintained code.
- `uv run ruff format <changed_python_files>`: format changed Python files.
- `uv run whnet-train --help`: inspect training options.
- `uv run whnet-train --max-steps 1`: perform a bounded GPU training check.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase constants. Add type hints to public interfaces. Volumes use `[N, C, D, H, W]`. Preserve FP32 physics and harmonic-loss calculations when changing mixed precision.

## Testing Guidelines

Use `unittest` and name files `test_*.py` under `tests/`. Cover numerical equivalence, checkpoint gradients, deterministic subject splits, and resume behavior when those areas change. Files under `experiments/` may plot or write outputs and are not regression tests.

## Commit & Pull Request Guidelines

Use short, descriptive commit subjects and keep commits focused. PRs should explain behavior changes, list validation commands, link relevant issues, and include convergence or timing evidence for training changes.

## Data & Runtime Practices

Never commit credentials, subject data, caches, or checkpoints. Split by source subject before augmentation. Check GPU occupancy before training and preserve active jobs. Use a new `artifacts/runs/` directory for every run and load current checkpoints with `whnet.training.load_trained_model()`.
