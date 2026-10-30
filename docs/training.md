Run joint training with:

```sh
uv run whnet-train --channels-last-3d
```

The defaults train both chi and harmonic proximal networks from fresh initialization
for 50 epochs. Both receive direct supervision from the first update. The harmonic
regularizer ramps over five epochs. A batch size of five and one accumulation
step give an effective batch size of five; GPU allocation is capped at 99% of
total device memory.

WHNet also adds `loss_chi_high`: for each target volume,
`M = 1[abs(chi_gt) > quantile(abs(chi_gt), 0.8)]`, with the quantile taken over
all its voxels. The loss is the full-volume mean of `M * abs(chi_gt-chi_pred)`
divided by that sample's masked target RMS; this makes its value insensitive to
a common scale change of target and prediction. `--lam-chi-high` weights this
term (new-run default `3.0`; set `0` to disable it).
Use `--no-normalize-high-chi-loss --lam-chi-high 100` to reproduce the previous
absolute-units term. Older runs retain their saved loss when resumed.

New runs draw 75% original and 25% synthetic samples per epoch, with repeated
records receiving independent views. Each sample's masked χ RMS is drawn from
a log-uniform 0.018–0.040 ppm core (95%) or a 0.040–0.080 ppm tail (5%). The
background source is scaled to a log-uniform local/background field RMS ratio
of 0.03–3. Validation keeps the previous fixed synthesis for comparable
subject metrics. Use `--chi-scale-mode legacy`, `--background-mode legacy`, and
`--synthetic-fraction -1` to restore the former data path. The model's input
normalization floor is 1e-6 for new runs; older checkpoints retain 1e-3 when
loaded through `load_trained_model()` or resumed.
For a new command-line run with `--init-ckpt`, the peak learning rate defaults
to `1e-5`; `--peak-lr` overrides it. Fresh initialization keeps `1e-4`, and
resume keeps the rate and scheduler saved in its checkpoint.

Each run creates a new `artifacts/runs/joint_<timestamp>/` directory containing:

- `config.json` and `split.json`: exact settings and subject identities.
- `metrics.jsonl`: losses, branch gradients, timing, memory and validation
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
modulo-56 mapping. Other datasets require `subjects.json` beside the tensors or
`--source-manifest subjects.json`, a JSON mapping from image filenames to subject
IDs. Validation uses one stored volume
per subject with three fixed augmentation/noise realizations.

To add the reproducible geometric cohort, run:

```sh
uv run python scripts/generate_geometric_data.py --seed 0
uv run whnet-train
```

The generator adds 651 `image_0651.pt`–`image_1301.pt` susceptibility tensors
and paired boolean masks to `data/training/joint/`. Every 160³ volume contains
4–12 randomly oriented ellipsoids, cylinders, cuboids, and cones with at least
one overlap and 10–20% union-mask occupancy. Each shape has a signed constant,
gradient, or smooth-texture susceptibility contribution; contributions add at
overlaps, and chi is zero outside the mask. The generator also writes
`synthetic_generation.json` and a complete `subjects.json` covering both
original and synthetic pairs. It refuses overwrites; `--resume` skips complete
pairs from the same configuration after an interrupted generation. Synthetic
identities use the `synthetic:` prefix and enter training only. Existing source
subjects retain the usual held-out validation split. Training automatically
reads `subjects.json` from the dataset directory when it exists; an explicit
`--source-manifest` path still takes precedence.
The training-start metrics report `synthetic_train_volumes` so each run records
how many generated pairs it uses. A bounded `--max-train-volumes` diagnostic
draws from both original and synthetic sources.
See the [generated data report](synthetic-data-report.md) for examples and
masked susceptibility comparisons with COSMOS and sim2.
The [training distribution report](training-distribution-report.md) compares
the actual augmented training samples, synthesized field ratios, and external
references. Rebuild it with
`MPLCONFIGDIR=/tmp/mpl-whnet .venv/bin/python -m scripts.report_augmented_training`.

To compare distribution changes at 60 iterations, start every run from the
same checkpoint and seed, and use a new run directory each time:

```sh
uv run whnet-train \
  --init-ckpt artifacts/runs/joint_20260929_202153/model_best.pth \
  --iter-start 60 --iter-max 60 --final-iters 60 \
  --seed 0 --peak-lr 1e-5 --max-steps 1
```

Remove `--max-steps 1` for a full candidate run. Use these overrides for
controlled ablations; keep the other settings identical:

| Candidate | Additional flags |
| --- | --- |
| Previous pipeline | `--chi-scale-mode legacy --background-mode legacy --synthetic-fraction -1 --no-normalize-high-chi-loss --lam-chi-high 100 --input-scale-floor 0.001` |
| χ mix only | `--background-mode legacy --no-normalize-high-chi-loss --lam-chi-high 100 --input-scale-floor 0.001` |
| χ and field ratio | `--no-normalize-high-chi-loss --lam-chi-high 100 --input-scale-floor 0.001` |
| χ, field ratio, normalized loss | New defaults |

Compare each run's physical-unit COSMOS and sim2 χ NRMSE relative to the
previous-pipeline run at the **same 60 iterations**. Prefer the candidate with
the smaller worst relative error among candidates whose held-out subject χ
NRMSE is at most 5% above that baseline. Report the fraction of clipped
updates as an optimization diagnostic. sim2 and COSMOS guide development and
checkpoint decisions here, so their final scores are development results.
After the ablations, run
`python -m scripts.compare_scale_runs <baseline-run> <candidate-run>...` to
apply this rule to each run's `model_best.pth` epoch. The comparator checks
the iteration count, source split, initializer, seed, and peak learning rate.

At the end of each completed epoch, WHNet also evaluates its EMA weights at
`final_iters` on the completed COSMOS SNR=100/background=10 simulation and on
sim2. COSMOS logs susceptibility and harmonic-background masked NRMSE; sim2
logs susceptibility masked NRMSE because it has no background reference. The
sim2 case uses centered padding of 20 voxels and its supplied fidelity-weight
map. Both cases enter model inference in physical units (ppm), with no
multiplicative factor applied during evaluation. Saved COSMOS solver-unit
inputs are restored to physical units once when loaded. The `--sim2-factor`
option remains for checkpoint compatibility; its default is now `1.0`, and
evaluation restores physical units even when an older run saved `0.264`.
These `external_*` metrics appear in
`metrics.jsonl` and, when enabled, Comet validation metrics. The held-out subject metric
`val_chi_k<final_iters>` still selects `model_best.pth`. Use
`--cosmos-simulation PATH`, `--sim2-factor`, and `--sim2-pad` to select inputs;
`--external-validation-every N` changes the frequency. Use
`--no-external-validation` to skip these benchmarks. Older checkpoints resume
with external evaluation disabled unless explicitly enabled.

Each external evaluation also saves comparison images under `images/` in the
run directory and uploads them to Comet when `--comet` is enabled. Every image
has axial, coronal, and sagittal slices of the ground truth, WHNet output, and
their difference. Susceptibility images use a fixed `[-0.1, 0.1]` ppm range
for all panels. The COSMOS harmonic-background image uses adaptive ranges.
sim2 has no harmonic-background reference, so it has only a susceptibility image.

Training synthesis applies sparse local-field perturbations to 10% of samples:
the field at one to three masked voxels is multiplied by an independent uniform
`X` between 10 and 100. In half of affected samples, each selected voxel also
has a six-connected masked neighbor multiplied by `-X`. These weights are applied
after noise is scaled from the clean local field and before the harmonic
background is combined with it. Validation synthesis remains unperturbed.
Use `--no-local-perturbations` to disable the augmentation. Checkpoints saved
before this option existed resume with it disabled unless explicitly enabled.

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
convolutions use BF16. Local metrics require no external service.

To register a training run with Comet, put `COMET_API_KEY=...` in the ignored
repository `.env` file or export `COMET_API_KEY` before launching. An exported
key takes precedence. Then run `uv run whnet-train --comet`. The run directory
records the Comet experiment key in `comet.json`; `--resume` continues that
experiment. Comet receives configuration and dataset details as parameters and
separate training and validation metrics with explicit steps and epochs. Manual
metric logging avoids duplicate console output and automatic model tracing.
`COMET_PROJECT_NAME` can override the default `wh-net-qsm` project, and Comet's
`COMET_WORKSPACE` environment setting selects the workspace.

Run regression checks with:

```sh
uv run python -m unittest discover -s tests -v
```

Checkpoint version markers prevent strict loading of new weights into the legacy
ADMM implementation. Exact source snapshots and PyTorch/CUDA versions are saved
with each new run. Gradients are clipped separately for the two branches.
