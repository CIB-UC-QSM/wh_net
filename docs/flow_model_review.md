# Flow model review and corrections

The current flow implementation keeps the original WH-Net physics loop and
replaces only its two learned prior operators with separately trained flow
networks. The reference is `EfficientADMMUnrolledNet.efficient_step` in
`src/whnet/model.py`; the new loop and FFT helper are in
`src/whnet/flow_admm.py`. Original WH-Net checkpoints and code are unchanged.

## Findings addressed

| Finding | Effect | Current implementation |
| --- | --- | --- |
| Joint data proximal solve used CG and two duals rather than WH-Net's measured-field split. | Different finite-depth dynamics; inner iterations and tolerances were additional parameters. | Three splits/duals, zero initialization, WH-Net update order, pointwise data split, FFT susceptibility, background average. |
| The solver returned the prior variables instead of the final physical variables. | At finite depth the reported maps need not be the estimates used in the physics update. | Return the final masked physical `chi,b`; trace their residuals. |
| A Fourier closed form with spatial weights would be invalid. | Mask and voxel weights do not commute with dipole convolution. | Isolate `W^2` in the measured-field split; divide only by `D^2+1` in the FFT solve. |
| Support projection can change the forward field. | Using the pre-projection field gives inconsistent subsequent background/dual updates. | Recompute the full padded dipole field after projecting susceptibility. |
| Historical weighting and background-penalty changes made old manifests unsafe. | Previously selected parameters are not transferable without validation. | Require ensemble v4 with solver identity; reject v2/v3 and retune using existing prior checkpoints. |
| Aggregate validation loss hid flow-time behavior. | A single loss curve did not show where the prior was improving. | Log endpoint MSE/NRMSE by fixed time and subject-mean aggregates. |
| Background denoising uses a target-interpolated input. | These plots alone can overstate total-field separation quality. | Label denoising panels and add measured-field-only `t=0` previews and separate metrics. |
| Scalar-only Comet logging and new experiments on resume prevented image histories. | Evolution could not be inspected consistently. | Fixed cases/noise/scales, stable image names, image step equal to epoch, saved experiment identifier and resume. |

The exact data objective remains
`0.5 ||W*(F^H D F chi+b-y)||_2^2`. There is no background Laplacian penalty in
that objective. `W` is a residual multiplier, not WH-Net's legacy precision
input: compare the loops using `legacy_weights = tau*W^2`.

The background branch remains a conditional total-field-to-background flow,
not a Gaussian prior. Only susceptibility uses Gaussian renoising. Network
architecture and `whnet-flow-v2` checkpoint shapes are preserved; the new
solver does not require retraining those priors.

## Remaining limitations

- The FFT susceptibility solve followed by masking matches optimized WH-Net,
  but is not an exact support-constrained minimizer. The pointwise and
  unconstrained Fourier substeps are closed forms; the projection is an
  approximation.
- Both targets use the measured-phase RMS. A strong background can make the
  normalized susceptibility small relative to unit Gaussian noise. Changing
  this scale requires consistent retraining and inference; it is not silently
  changed here. NRMSE and time-resolved curves help expose poor relative fit.
- Learned velocity fields are not certified proximal operators. Neither this
  three-split scheme nor the conditional background adaptation automatically
  inherits the paper's fixed-point guarantees. Inspect split residuals and
  repeated-seed stability.
- Harmonicity is learned from background targets, not enforced exactly by the
  U-Net. Interior Laplacian metrics are diagnostics only.
- The `t=0` background image is one endpoint prediction, not an ODE-integrated
  flow or coupled reconstruction. Denoising panels at other times use targets
  in the interpolation; ensemble validation remains necessary.
- Five-epoch priors and one-subject tuning are preliminary. Previous COSMOS
  and sim2 NRMSE values belong to the historical solver, not this version.

## Verification and next run

Regression tests compare the FFT/pointwise physics against independently
assembled dense subproblems and compare five full iterations with optimized
WH-Net under identical projected priors and nonbinary residual weights.
Additional tests cover seeded inference, odd grids, masks, checkpoint
compatibility, background conditioning, deterministic training/resume, fixed
image inputs, real PNG creation, Comet image steps, and experiment resumption.
Comet calls are tested with an SDK test double; this does not certify an
authenticated upload to a live account.

Validation: `.venv/bin/python -m unittest discover -s tests -v` passes all
38 tests, including two-epoch CPU checks for both branches. `ruff check src
tests scripts` and `git diff --check` also pass.

Train with `--comet --image-every 1 --image-examples 1` to create new epoch
histories. In the Image Panel, select a stable image name and use its stepper;
image steps correspond to epochs. See the [logging guide](flowadmm.md#epoch-images-and-comet)
for commands and the [architecture note](flow_model_architecture.md) for the
equations. Retune the existing priors with `whnet-flow-tune`, then run
`whnet-flow-external-evaluate` on COSMOS/sim2 before claiming new reconstruction
scores. No new external benchmark is claimed by this code change.
