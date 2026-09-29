# WH-Net

WH-Net is a PyTorch implementation of learned unrolled ADMM for quantitative susceptibility mapping (QSM). It jointly estimates susceptibility (chi) and the harmonic background field from a measured phase volume.

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

For independently trained susceptibility and harmonic-background flow priors coupled by FlowADMM, see [docs/flowadmm.md](docs/flowadmm.md).
The flow solver uses `0.5 * ||W * (F^H D F chi + b - y)||_2^2`, with `W` as
residual multipliers and `W^2` in its pointwise measured-field split. Its three
ADMM splits follow WH-Net, using an FFT susceptibility solve and pointwise
background update without CG. Flow training supports epoch-indexed Comet
image histories; see [the logging guide](docs/flowadmm.md#epoch-images-and-comet).

Train the adaptive WH-TV parameter predictor on the same data with:

```sh
uv run wh-tv-train --comet
```

It predicts a spatial noise-variance map and all five WH-TV parameters, then reconstructs with a 2,000-iteration solve. See [docs/adaptive-wh-tv.md](docs/adaptive-wh-tv.md) for gradient, logging, resume, and inference details.

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
