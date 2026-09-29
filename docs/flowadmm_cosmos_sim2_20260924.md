# Historical FlowADMM COSMOS and sim2 evaluation — 2026-09-24

This report describes the superseded version 2 data step. It interpreted `W`
as fidelity precision and added a Laplacian penalty to the background inside
the data objective. The current version 4 uses only
`0.5 * ||W * (F^H D F chi + b - y)||_2^2`, where `W` multiplies the residual
and the measured-field split uses `W^2`. It also replaces the former joint
CG solve with WH-Net's three-split closed-form updates. The settings, images,
and scores below are preserved as historical results and have not been
recomputed with these changes. The existing v2 prior checkpoints remain compatible, but ensemble
tuning and external evaluation must be rerun in fresh directories. Although
both external cases used binary masks (`W^2 = W`), removing the extra penalty
still changes their solver, and the nonbinary weights used in validation also
change the parameter-selection objective.

The independently trained five-epoch susceptibility and conditional-background priors were assembled with `whnet.flow_admm.load_flow_run`. A coordinate search on **one held-out source subject, one deterministic synthesis view** selected 30 ADMM iterations, `tau=50`, harmonic weight `1`, flow time `0.1→0.95`, and susceptibility sample schedule `1/2/5`. The search did not use COSMOS or sim2 references. The validation susceptibility NRMSE was `1.0495` and background NRMSE was `0.0125`; this small validation sample limits confidence in the selected settings.

Masked NRMSE is `||mask * (reconstruction - reference)||₂ / ||mask * reference||₂`, matching the existing external evaluator. The reported images and NIfTI volumes are in physical units after the same dataset factors and cropping used by the existing WHNet evaluations. Both cases use the binary mask as fidelity weights.

| Simulation | Susceptibility NRMSE | Background NRMSE | Reconstruction images |
| --- | ---: | ---: | --- |
| COSMOS SNR 100, BG 10 | 0.7510 | 0.0290 | [χ comparison](../artifacts/runs/flow_external_20260924_5epoch_extended/cosmos/chi_comparison.png), [background comparison](../artifacts/runs/flow_external_20260924_5epoch_extended/cosmos/background_comparison.png) |
| sim2 | 0.9812 | unavailable | [χ comparison](../artifacts/runs/flow_external_20260924_5epoch_extended/sim2/chi_comparison.png), [background estimate](../artifacts/runs/flow_external_20260924_5epoch_extended/sim2/background_reconstruction.png) |

The corresponding [ensemble manifest](../artifacts/runs/flow_ensemble_20260924_extended/ensemble.json) and [evaluation metrics](../artifacts/runs/flow_external_20260924_5epoch_extended/metrics.json) record checkpoint paths, parameters, data provenance, and inference diagnostics. Each case directory also contains a per-iteration solver trace and reference/reconstruction/error NIfTI volumes. sim2 has no background reference, so no background NRMSE is reported.

These are preliminary reconstructions. The COSMOS susceptibility image underestimates some structures, and sim2 has pronounced error despite visible anatomy. The inner conjugate-gradient solve reached its 50-step cap on both cases; its final relative residual was `0.00166` for COSMOS and `0.01659` for sim2, above the nominal `1e-5` tolerance. A sim2 diagnostic with 200 inner iterations reduced that residual to `0.000243` but gave susceptibility NRMSE `0.9918`, so inner-solve accuracy alone did not resolve the reconstruction error. The five-epoch priors and one-subject parameter selection should not be treated as a final benchmark.
