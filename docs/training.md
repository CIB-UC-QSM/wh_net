Run joint training with:

```sh
uv run whnet-train --channels-last-3d
```

The defaults train both chi and harmonic proximal networks from fresh initialization
for 50 epochs. Both receive direct supervision from the first update. The harmonic
regularizer ramps over five epochs. A batch size of one and five accumulation
steps give an effective batch size of five; GPU allocation is capped at 25% of
total device memory to coexist with the existing job.

Each run creates a new `artifacts/runs/joint_<timestamp>/` directory containing:

- `config.json` and `split.json`: exact settings and subject identities.
- `metrics.jsonl`: losses, branch gradients/updates, timing, memory and validation
  error at 20, 30 and 55 iterations.
- `model_best.pth`: EMA weights selected by subject-mean chi NRMSE at 55 iterations.
- `model_last.pth`: most recent live weights.
- `training_last.pt`: model, EMA, optimizer, schedule and resume position.
- `status.json`: running, paused, failed or complete.

Use `--resume <run>/training_last.pt` to continue. SIGINT/SIGTERM requests a
checkpoint after the current optimizer update. `--max-steps N` stops after N total
optimizer updates; set `--max-steps 0` when resuming a bounded diagnostic run.

The split groups all augmentations of a subject. Images 0000–0314 use the
user-confirmed source index modulo 105; images 0315–0650 follow the generator's
modulo-56 mapping. Other datasets require `--source-manifest subjects.json`, a JSON
mapping from image filenames to subject IDs. Validation uses one stored volume
per subject with three fixed augmentation/noise realizations.

Original tensor files are preserved. Compact FP32 images and boolean masks /
dilations are cached in `artifacts/cache/training/`. Cache keys include source paths, sizes
and modification times.

New checkpoints use corrected projected physics, input scaling and normalized,
bounded proximal conditioning. Load them with the matching implementation:

```python
import torch
from whnet.training import load_trained_model

model = load_trained_model("<run>/model_best.pth", device="cuda")
# phase is in physical units; normalization and output rescaling are internal.
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    chi, background = model(phase, mask, dipole, weights)
```

Use `--init-ckpt artifacts/checkpoints/checkpoints_scratch7/model_best.pth` for an explicit warm-start.
This initializes both branches and re-enables harmonic optimization, but the old
checkpoint already saw subjects in the new validation split. Such validation is
a fine-tuning diagnostic, not a clean estimate on unseen subjects.

`--compile-proximal`, `--channels-last-3d` and their `--no-...` counterparts allow
performance comparisons; both are enabled by default on CUDA. FFTs, losses and reductions stay in FP32; eligible CUDA
convolutions use BF16. Local metrics require no external service; `--comet`
optionally enables the existing Comet integration.

Run regression checks with:

```sh
uv run python -m unittest discover -s tests -v
```

Checkpoint version markers prevent strict loading of new weights into the legacy
ADMM implementation. Exact source snapshots and PyTorch/CUDA versions are saved
with each new run. Gradients are clipped separately for the two branches.
