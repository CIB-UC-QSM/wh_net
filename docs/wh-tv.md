# WH-TV reconstruction

`whnet.wh_tv.wh_tv` implements linear weak-harmonic weighted total variation
reconstruction on PyTorch tensors. Its numerical formulation follows the FANSI
`WH_wTV.m` implementation by Carlos Milovic and Patrich Fuchs, revision
`4371ba3ecf3831af46dc7934cbaf7d32f7a7b16f`.

The solver accepts measured phase, a Fourier dipole kernel, and an optional mask,
all on the same device. Volumes use `[N, 1, D, H, W]`. It returns susceptibility,
harmonic background, iteration count, and the last percentage update. The
`weight` argument is squared inside the data term, matching the MATLAB solver.
Spatial differences are periodic and use unit voxel spacing. Its default
initialization does not invoke the separate MATLAB MSMV preconditioner. A supplied
`initial_solution` warm starts the primal variables with zeroed dual variables;
it does not resume the ADMM state. The final iteration updates susceptibility
only, so the returned background comes from the preceding split update.

For reproducible comparisons, `scripts/optimize_wh_tv.py` tunes parameters on
the COSMOS simulation; `scripts/evaluate_wh_tv.py` evaluates the saved run;
`scripts/benchmark_whnet_wh_tv.py` compares WH-TV with WHNet. Graz evaluation
uses `scripts/benchmark_graz.py`. The adaptive WH-TV model and its training
commands are documented in [adaptive-wh-tv.md](adaptive-wh-tv.md).
