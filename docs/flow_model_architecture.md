# Flow model architecture and WH-Net closed-form updates

This note describes the implemented FlowADMM reconstruction in
`src/whnet/flow_admm.py`. It is specific to this repository's masked 3D QSM
formulation: a susceptibility map `chi` and a harmonic background field `b`
are reconstructed jointly from a measured total field `y`.

## Overview

The model combines three components:

1. Two separately trained, frozen flow priors provide learned updates for
   `chi` and `b`.
2. A physics data step enforces that the dipole field from `chi`, plus `b`,
   explains the measured field inside the mask.
3. ADMM-style dual updates couple the learned prior outputs to the physics
   solution over several outer iterations.

```text
                         measured total field y, mask M, dipole D
                                            |
                                            v
        +-------------------------------------------------------+
        |  measured-field split s: pointwise weighted update    |
        +-------------------------------------------------------+
                         |                         |
                         v                         v
       Gaussian-renoise chi flow           conditional background flow
        stochastic Monte Carlo average       measured-field conditioned
                         |                         |
                         +---------- priors (u, v) ------------+
                                            |
                                            v
                         FFT chi solve -> mask -> field -> b average
                                            |
                                            v
                                  three dual-variable updates
                                            |
                                      next outer iteration
```

All volumes have shape `[N, 1, D, H, W]`. Before reconstruction, the measured
phase is divided by its masked RMS. This same scale is used during prior
training and is restored at the final output, so the learned priors see a
consistent numerical range while saved reconstructions retain physical units.

## Flow-prior architecture

`FlowVelocity3D` is a compact 3D U-Net velocity network. Its base width is
`base_channels` (eight by default) and the encoder widths are `w`, `2w`,
`4w`, and `8w`.

```text
input channels
    |
    v
FlowBlock(w) -- average pool --> FlowBlock(2w) -- pool --> FlowBlock(4w)
                                                              |
                                                            pool
                                                              v
                                                        FlowBlock(8w)
                                                              |
                                               trilinear upsample + skip
                                                              v
                                                        FlowBlock(4w)
                                                              |
                                               trilinear upsample + skip
                                                              v
                                                        FlowBlock(2w)
                                                              |
                                               trilinear upsample + skip
                                                              v
                                                        FlowBlock(w)
                                                              |
                                                         1x1x1 convolution
                                                              v
                                                     scalar velocity v_t
```

Each `FlowBlock` applies two padded `3x3x3` convolutions, GroupNorm, and SiLU.
The second normalized activation is modulated with FiLM scale and shift
parameters generated from the flow time. Flow time is encoded by 16 sinusoidal
frequencies (sine and cosine, 32 values total), then passed through an MLP with
width 128. This gives the same U-Net different behaviour at different points
along its interpolation path. The final `1x1x1` output layer is initialized to
zero, so a new network initially implements the identity denoiser rather than
an arbitrary large update.

### Susceptibility branch

The susceptibility velocity model receives two channels:

```text
[chi_t * M, M]
```

During training, it uses the Gaussian interpolation

```text
chi_t = M * (t * chi + (1 - t) * epsilon),   epsilon ~ Normal(0, I),
```

and learns the velocity `chi - epsilon`. At reconstruction time, a current
proposal is renoised with fresh Gaussian draws and denoised by

```text
D_t(chi_t) = M * (chi_t + (1 - t) * v_chi(chi_t, t, M)).
```

Several independent draws can be averaged. The sample count can increase late
in the outer iterations to reduce Monte Carlo variability when the solution is
near its final state.

### Background branch

The background field is not treated as Gaussian noise. It is trained to
separate a background field from the measured *total* field. Its velocity model
therefore receives three channels:

```text
[b_t * M, y * M, M]
```

and uses the conditional interpolation

```text
b_t = M * (t * b + (1 - t) * y).
```

It learns `b - y`, with `y` retained as an explicit conditioning input. Its
reconstruction update is deterministic:

```text
D_t(b_t, y) = M * (b_t + (1 - t) * v_b(b_t, t, M, condition=y)).
```

This distinction is important: adding random Gaussian noise to this branch
would train a generic background generator, not a map from the observed total
field to the background component that must be removed from it.

## WH-Net physics splitting

Let `K = F^H D F` denote the dipole forward operator: Fourier transform `F`,
pointwise dipole multiplication `D`, then inverse transform `F^H`. The data
consistency term at every outer iteration is exactly

```text
f(chi, b) = 1/2 || W * (K chi + b - y) ||_2^2.
```

`W` is a real, nonnegative voxelwise **residual multiplier**. Therefore its
square, `W^H W = W^2`, appears in the measured-field update. It is
not a precision supplied directly to the solver. For example,
`W=0.5` gives a squared-error coefficient of `0.25`.

The implementation uses binary support `M` and sets `W` to zero outside it.
It follows `EfficientADMMUnrolledNet`: `chi` is projected onto `M`, but the
dipole field `K chi` is computed on the full padded volume. Background `b`
is retained on that full volume internally; the prior sees a masked proposal,
and both final output volumes are masked. Masking the dipole field prematurely
would change the split constraints because `M K` and `K M` do not commute.

The background flow supplies the learned background regularization. There is
no Laplacian penalty in the data problem. `harmonic_rms` remains a
diagnostic measured with an interior mask `I`: it is one where a complete
`3x3x3` Laplacian stencil lies inside `M`. This avoids evaluating zero-padding
artifacts at the boundary. If there is no complete stencil, the metric is zero.

Introduce three auxiliary variables and scaled duals:

```text
s = K chi + b       dual eta_s   (measured total field)
u = chi             dual eta_u   (susceptibility prior)
v = b               dual eta_v   (background prior)
rho = 1 / tau       same penalty for all three constraints
```

Formally the split objective is `0.5 ||W*(s-y)||^2 + R_chi(u) + R_b(v;y)`.
The learned flow operators replace explicit proximal maps of `R_chi` and
`R_b`; this notation does not imply that either network is the proximal map of
a known convex regularizer. Eliminating the equality constraints recovers the
requested weighted data term without adding a harmonic penalty.

### One outer iteration

Initialize `chi`, `b`, and all three duals to zero, as in WH-Net. Using the
previous physical estimates, apply the following order:

```text
1. s = (tau*W^2*y + K chi + b + eta_s) / (1 + tau*W^2)
2. u = Gaussian-renoise/denoise mean of M*(chi + eta_u)
   v = conditional total-field flow of M*(b + eta_v), conditioned on y
3. chi_raw = F^-1[(D*F(s - eta_s - b) + F(u - eta_u)) / (D^2 + 1)]
   chi = M * chi_raw
4. field = K chi                       # recompute AFTER projection
   b = 0.5 * (s - eta_s - field + v - eta_v)
5. eta_s += field + b - s
   eta_u += chi - u
   eta_v += b - v
```

The data split in step 1 exactly minimizes
`0.5 ||W*(s-y)||^2 + rho/2 ||s-(K chi+b+eta_s)||^2`, even for spatially
varying weights. Step 3 solves the *unconstrained padded-grid* quadratic
`rho/2 ||K chi-(s-eta_s-b)||^2 + rho/2 ||chi-(u-eta_u)||^2` by diagonalizing
the real dipole operator. The common `rho` cancels; hence the denominator is
`D^2+1`, not `D^2+tau*W^2`. Step 4 averages the two background proposals.

The susceptibility support projection is the same approximation used by the
maintained optimized WH-Net. In general, solving on the padded grid and then
masking is **not** the exact solution of the mask-constrained quadratic.
Closed forms here therefore mean exact unconstrained FFT and pointwise
substeps, followed by WH-Net's support projection—not an exact joint masked
proximal solve. This also prevents an incorrect shortcut that divides by a
Fourier expression containing spatially varying `W` or `M`.

At termination, return the physical estimates `M*chi` and `M*b`, restored to
input units, rather than the last prior outputs `u` and `v`. Those quantities
may differ noticeably at finite iteration depth.

### Why CG is no longer needed

Version 3 solved a joint, support-constrained data proximal problem with
conjugate gradient. That was a different splitting from WH-Net. Restoring its
measured-field auxiliary variable isolates arbitrary voxelwise weights in a
pointwise update; the remaining susceptibility update has a fixed Fourier
denominator and the background update is a pointwise average. There is no
inner linear iteration, tolerance, or CG iteration cap in the current solver.
All physics stays FP32, including when the flow networks use BF16 autocast.

With identical proximal outputs, the new iteration matches
`EfficientADMMUnrolledNet` after mapping its precision input to `tau*W^2`.
Its original U-Nets and training are otherwise unchanged. The flow version
keeps flow-time FiLM, Gaussian Monte Carlo averaging for susceptibility, and
measured-field conditioning for background; these are intentional differences.

### Diagnostics

The trace records each of the three split residuals, their combined
`primal_rms`, updates of both physical variables, `data_rms`,
`weighted_data_rms`, and `harmonic_rms`. RMS values are measured inside `M` in
phase-normalized units. In these coordinates,
`f = 0.5 * voxel_count * weighted_data_rms^2`.
Harmonic residuals never enter the solver equations. Small split residuals are
an empirical diagnostic, not a convergence guarantee for these learned priors.

## Compatibility with previous runs

Current manifests use `whnet-flow-ensemble-v4`,
`data_term: weighted_residual_l2`, and `solver_type: whnet_closed_form`.
Version 2 used an incorrect objective; version 3 used the previous CG splitting.
Both are rejected so their tuned parameters cannot silently be reused with a
different solver. Retune in a fresh directory using the existing
`whnet-flow-v2` susceptibility and conditional-background checkpoints. Their
independent flow-matching loss is unchanged. Historical COSMOS/sim2 scores
require fresh evaluation after retuning and are not current solver results.

See [training and Comet image history](flowadmm.md#epoch-images-and-comet)
and [review findings and remaining limitations](flow_model_review.md).
