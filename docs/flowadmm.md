# FlowADMM priors for WH-Net

This implementation adapts the [FlowADMM renoise-denoise operator](https://arxiv.org/html/2605.08640v1) to joint 3D susceptibility (`chi`) and harmonic background (`background`) reconstruction. The two velocity networks are trained independently, frozen, and used as the two prior steps of a coupled ADMM solver. The background branch is a conditional total-field-to-background flow, rather than a Gaussian generative prior. Existing `whnet-train` checkpoints and inference remain separate.

For the U-Net layout, branch-specific flow paths, and derivation of the
WH-Net closed forms, see [Flow model architecture](flow_model_architecture.md).
The [model review](flow_model_review.md) records corrected issues and remaining
limitations.

## Model and data step

The susceptibility prior uses `chi_t = t chi + (1-t) epsilon`, with masked standard Gaussian `epsilon`, and regresses `chi - epsilon`. The background branch uses `b_t = t b + (1-t) y`, where `y` is the measured total field, and regresses `b - y`. It also receives `y` as a conditioning channel. Both use `D_t(q_t) = q_t + (1-t) v(q_t,t,condition,mask)`. The same per-volume masked RMS of measured phase scales both targets and reconstruction variables.

Let `K = F^H D F` be the dipole forward operator, with `F` the Fourier
transform, `F^H` its inverse, and `D` the dipole kernel. The solver uses the
data consistency term

```text
f(chi,b) = 0.5 ||W * (K chi + b - y)||_2^2.
```

Here `y` is measured phase and `W` multiplies the residual directly. Inputs
`weights` are interpreted as real, nonnegative residual multipliers and are
zeroed outside the mask. Consequently, the measured-field split uses `W^2`.
For example, a voxel weight of `0.5` contributes `0.25` times its squared
residual. Keep the dipole field on the full padded grid in the split equations;
masking the field there is not interchangeable with masking susceptibility.

The solver now follows WH-Net's three splits, update order, zero initialization,
and three scaled duals. With equal split penalty `rho=1/tau`, each iteration
uses:

```text
s = (tau*W^2*y + K chi + b + eta_s)/(1 + tau*W^2)
u = mean_chi_flow(M*(chi + eta_u))
v = conditional_background_flow(M*(b + eta_v), y)
chi = M*F^-1[(D*F(s-eta_s-b) + F(u-eta_u))/(D^2+1)]
field = K chi
b = 0.5*(s-eta_s-field+v-eta_v)
eta_s += field+b-s; eta_u += chi-u; eta_v += b-v
```

There is no CG inner loop or CG configuration. The FFT solve is exact on the
unconstrained padded grid; its following support projection is WH-Net's
approximation, not an exact support-constrained solve. Background is retained
on the full grid internally and masked at output. Physics stays FP32. The
returned maps are the final physical `chi,b`, not the last prior outputs.
The data objective contains only the weighted field residual. Background
regularization comes from the conditional flow; interior Laplacian errors are
diagnostics, without contributing a penalty.

The susceptibility branch averages Gaussian-renoise outputs, with default
sample counts 1, 2, and 5 over the first 50%, next 30%, and final 20% of
iterations. Background uses one deterministic conditional update. The default
time schedule rises from 0.3 toward 0.95 over 30 iterations. These are starting
values for validation tuning, not universal QSM settings.

## Train both priors

Run from the repository root after checking GPU occupancy. Both commands use the same source-subject split when their data, manifest, validation fraction, and seed match. Each command creates a fresh run directory under `artifacts/runs/` by default.

```sh
uv run whnet-flow-train --branch chi
uv run whnet-flow-train --branch background
```

Add `--comet` to log configuration, step/epoch metrics, and epoch images to the
`wh-net-qsm` Comet project. Experiments are named after the run directory and
tagged `flow-chi` or `flow-background`. Local `metrics.jsonl` and PNG logging
remain active with or without Comet. Configure credentials using your existing
Comet environment/configuration; credentials are not stored in the run.

The defaults are 50 epochs, batch size one, five accumulation steps, AdamW
with peak learning rate `1e-4`, and EMA validation. Fixed validation times are
0.2, 0.5, 0.8, and 0.95; checkpoint selection still minimizes subject-mean
masked endpoint MSE. Logging also records subject-mean NRMSE, MSE/NRMSE by
flow time, gradient norm/clipping, epoch mean loss, and epoch duration. The
background run records interior Laplacian MSE and a separate measured-field-only
`t=0` endpoint MSE/NRMSE diagnostic. This one-step separation preview is not
the full ensemble reconstruction and does not select checkpoints.

Both branches use existing `FieldSynthesizer` targets and a source-safe split.
Independent training fits velocity targets, not the ensemble data objective,
so existing `whnet-flow-v2` priors do not require retraining for the solver
change. For another dataset, pass `--data-dir` and `--source-manifest` to both
runs.

Use `--max-steps 1 --device cpu --num-workers 0 --base-channels 4` for a bounded diagnostic. Resume a paused run with `--resume artifacts/runs/<run>/training_last.pt --branch <branch>` and the same data/configuration arguments. The run contains `split.json`, `config.json`, `metrics.jsonl`, `training_last.pt`, and, after the first validation, `model_best.pt`.

### Epoch images and Comet

Images are logged after each completed epoch by default:

```sh
uv run whnet-flow-train --branch chi --comet --image-every 1 --image-examples 1
uv run whnet-flow-train --branch background --comet --image-every 1 --image-examples 1
```

Each selected fixed validation realization gets orthogonal source, flow-input,
reference, predicted-endpoint, and error views at every validation time. The
background branch also logs a `total_field` series whose model input contains
only the measured field. Validation subjects, synthesis seeds, Gaussian draws,
and reference-derived display scales stay fixed across epochs. Ordinary
denoising panels use target-interpolated inputs and are labelled accordingly;
they are not evidence of full inversion quality.

Image names are stable, for example `validation_chi_example00_t050`.
`log_image` receives `step=epoch` and `overwrite=False`; epoch and optimizer
step are also metadata. Metrics retain optimizer-step indexing and their epoch
field. In Comet, add an **Image Panel**, select that image name, and use its
**Stepper/slider** to move through epochs. This follows the documented
[image-name and step grouping](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/image-panel/)
and [Python image API](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#log_image).

Local copies are `images/<name>/epoch_0001.png` with matching JSON metadata.
`--image-every N` changes cadence; `--image-examples 0` disables image creation.
A run's `comet.json` saves only its experiment identifier. Resuming with
`--comet` reopens that same experiment using
[ExistingExperiment](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/ExistingExperiment/),
so its image history continues. Older runs without that file start a new
experiment; historical images cannot be recovered automatically. Image cadence
and example count can change on resume without altering training.

## Tune and load the ensemble

```sh
uv run whnet-flow-tune \
  --chi-checkpoint artifacts/runs/<chi-run>/model_best.pt \
  --background-checkpoint artifacts/runs/<background-run>/model_best.pt
```

Tuning requires identical subject splits for the two priors. It searches the
inverse split penalty `tau`, initial flow time, time exponent, susceptibility
sample schedule, and 20/30/55 iteration depths in a fixed coordinate search.
Each candidate uses the same validation realizations and seed. `search.json`
records all candidates; `ensemble.json` records the selected settings, solver
identity, flow evaluation counts, subject-mean susceptibility/background NRMSE,
unweighted/weighted field residuals, and diagnostic harmonic residual. Selection
minimizes susceptibility NRMSE, breaking ties with background NRMSE.

Current ensembles use `format: whnet-flow-ensemble-v4`,
`data_term: weighted_residual_l2`, and `solver_type: whnet_closed_form`.
Version 2 used an incorrect objective and version 3 used the CG splitting.
The loader rejects both with instructions to rerun `whnet-flow-tune`. Keep the
trained v2 priors and create a fresh tuning run. Previously reported settings
and reconstruction scores are historical, not results of this solver.

```python
from whnet.flow_admm import load_flow_run

model = load_flow_run("artifacts/runs/<ensemble-run>/ensemble.json", device="cuda:0")
chi, background, trace = model(phase, mask, dipole, weights, seed=0, return_diagnostics=True)
```

Inputs and outputs are `[N, 1, D, H, W]`. `dipole` may use a full or real-FFT
grid. Pass residual weights `W` directly. The seed fixes Monte Carlo draws.
`trace` contains the three split residuals, combined primal residual, physical
update sizes, `data_rms`, `weighted_data_rms`, and diagnostic `harmonic_rms`;
there are no CG fields. Trace residuals are phase-normalized; evaluators
rescale them into input units. The interior mask is only for the harmonic
metric, which is zero without a complete stencil. Compare methods only on
test subjects unseen by both pipelines.

The paper's fixed-point guarantee does not directly cover this masked, two-prior QSM solver, especially its conditional background branch. Use the recorded traces and repeated-seed evaluations to assess empirical stability. Version 1 Gaussian-background checkpoints are intentionally incompatible with this formulation.

## Held-out comparison

When an independent test set is available, evaluate both methods on exactly the same subjects and deterministic synthesis views:

```sh
uv run whnet-flow-evaluate \
  --ensemble artifacts/runs/<ensemble-run>/ensemble.json \
  --baseline artifacts/runs/<joint-run>/model_best.pth \
  --data-dir <unseen-source-volumes> \
  --source-manifest <filename-to-subject-json>
```

The command rejects any test subject appearing in either prior's or the baseline's saved training/validation split. It writes subject-mean errors, residuals, and synchronized seconds per realization to a new JSON report.

## COSMOS and sim2 simulations

After tuning on the source-subject validation split, keep the ensemble fixed for the external simulations:

```sh
uv run whnet-flow-external-evaluate \
  --ensemble artifacts/runs/<ensemble-run>/ensemble.json
```

The command reuses the existing COSMOS SNR=100/background=10 simulation and legacy sim2 crop, scaling, padding, and continuous dipole kernel. Fidelity weights are the binary evaluation mask for both cases. It saves physical-unit NIfTI volumes, orthogonal reference/reconstruction/error PNGs, per-case solver traces, and masked NRMSE to a new `artifacts/runs/flow_external_*/` directory. COSMOS has both susceptibility and background references; sim2 has only a susceptibility reference, so its background reconstruction is shown without an NRMSE. Use `--run-dir` to set an explicit fresh output directory, or `--datasets cosmos` / `--datasets sim2` to run one case. The COSMOS and sim2 targets are never used for ensemble parameter selection.
