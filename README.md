# WHNet and WH-TV

This repository contains two approaches to quantitative susceptibility mapping (QSM): WHNet, a learned unrolled ADMM model, and WH-TV, a weighted total variation solver. Both estimate susceptibility (chi) and the harmonic background field from a measured phase volume. An adaptive WH-TV model predicts its solver parameters from the input.

## Repository layout

- `src/whnet/`: maintained model, data, physics, and training code.
- `tests/`: deterministic regression tests for numerical behavior and resume support.
- `scripts/`: data-preparation entry points.
- `experiments/`: exploratory evaluations, diagnostics, and archived training code.
- `docs/`: training instructions and experiment reports.
- `data/`: local datasets and evaluation inputs; contents are ignored by Git.
- `artifacts/`: checkpoints, caches, logs, and generated outputs; contents are ignored by Git.

## Setup

Install the locked Python 3.12 environment:

```sh
uv sync
```

Run the maintained test suite and lint checks:

```sh
uv run python -m unittest discover -s tests -v
uv run ruff check src tests scripts
```

## Training

Inspect all options or start a run:

```sh
uv run whnet-train --help
uv run whnet-train
```

Training defaults to `data/training/joint/`, caches decoded volumes under `artifacts/cache/training/`, and writes new runs to `artifacts/runs/joint_<timestamp>/`. See [docs/training.md](docs/training.md) for checkpoint and resume details.

Train the adaptive WH-TV parameter predictor on the same data with:

```sh
uv run wh-tv-train --comet
```

It predicts a spatial noise-variance map and all five WH-TV parameters, then reconstructs with a 2,000-iteration solve. See [docs/adaptive-wh-tv.md](docs/adaptive-wh-tv.md) for gradient, logging, resume, and inference details.

For direct WH-TV reconstruction and WHNet comparisons, see [docs/wh-tv.md](docs/wh-tv.md).

## Inference

```python
from whnet.training import load_trained_model

model = load_trained_model(
    "artifacts/checkpoints/checkpoints_joint_20260908_001534/model_best.pth",
    device="cuda",
)
chi, background = model(phase, mask, dipole, weights)
```

Legacy checkpoints require `whnet.model.ADMMUnrolledNet`. Exploratory examples live under `experiments/evaluation/`.
