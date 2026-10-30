# Adaptive WH-TV training

The adaptive WH-TV model predicts a relative noise-variance volume and all five
WH-TV solver parameters from the measured phase and mask. Its U-Net decoder
produces the spatial variance while an MLP attached to the bottleneck predicts
`alpha1`, `beta`, `mu1`, `mu2`, and `muh`.

Start a production run with:

```sh
uv run wh-tv-train --comet
```

The default solve uses 2,000 WH-TV iterations. The complete solve determines
the reconstruction and loss, but autograd is retained only for the final 25
iterations. This last-K approximation keeps the long forward solve while
bounding training memory. Production configurations reject fewer than 1,000
iterations.

The command uses the same `data/training/joint/` volumes, subject-level split,
field synthesis, validation realizations, and cache as WHNet. Every run creates
a new `artifacts/runs/adaptive_wh_tv_<timestamp>/` directory containing the
configuration and split, JSONL metrics, source snapshots, local validation
images, live/EMA-selected model weights, resumable optimizer state, and status.

The shared synthesizer samples masked-RMS SNR uniformly from 40 to 100,
draws exterior-source amplitudes uniformly from 0.01 to 20 outside the
7-voxel-dilated mask, and uses the same heteroscedastic reliability/noise model
for both WHNet and adaptive WH-TV. These fixed conditions are written to
`environment.json` and the training-start metrics (including Comet).
Both training pipelines also use the sparse, training-only local-field
perturbations described in [training.md](training.md); validation stays clean.
Use `--no-local-perturbations` to disable them.

At every validation epoch, adaptive training also evaluates the EMA model on
the same external cases used for WHNet:

- COSMOS: the newest completed SNR=100/background=10 saved simulation.
- sim2: 20-voxel centered padding, the mask bounding-box crop, and the
  continuous dipole kernel from the legacy sim2 workflow.

Both training benchmarks now pass physical-unit phase and χ to the model.
Saved COSMOS solver-unit inputs are restored to ppm once before inference;
sim2 uses physical units directly by default. The legacy `--sim2-factor`
option remains accepted for checkpoint compatibility, with physical-unit
evaluation regardless of its value.

COSMOS susceptibility and background NRMSEs are logged. sim2 logs
susceptibility NRMSE only because that dataset has no background ground truth.
The matching reference/reconstruction/error figures are saved locally and sent
to Comet. Use `--no-external-validation` only for bounded diagnostics, or
change `--external-validation-every` to reduce benchmark frequency.

Run both benchmarks independently against a selected checkpoint with:

```sh
uv run wh-tv-evaluate \
  --checkpoint artifacts/runs/<run>/model_best.pth \
  --comet
```

This writes per-dataset `metrics.json`, WHNet-style comparison PNGs, and
NIfTI reference/reconstruction/error volumes under a new
`artifacts/runs/adaptive_wh_tv_external_<timestamp>/` directory.

Resume an interrupted run with:

```sh
uv run wh-tv-train --resume artifacts/runs/<run>/training_last.pt
```

When `--comet` is enabled, the run logs component and total losses,
susceptibility/background NRMSE, predicted parameter and variance statistics,
runtime and memory metrics, and a fixed validation figure after every epoch.

Load the selected model with:

```python
import torch
from whnet.adaptive_wh_tv_training import load_adaptive_wh_tv

model = load_adaptive_wh_tv("<run>/model_best.pth", device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(phase, mask, dipole)

chi = output.chi
background = output.background
relative_variance = output.variance
parameters = output.parameters.as_dict()
```

For bounded tests only, combine a short horizon with
`--allow-short-solver`. Such runs are diagnostics and are not valid training
configurations.
