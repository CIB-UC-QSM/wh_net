# WH-Net QSM

Physics-informed unrolled ADMM for joint susceptibility and weak-harmonic
background-field reconstruction.

## Training

Run training from the repository root:

```bash
.venv/bin/python train_console.py
```

Training initializes the checkpoint-compatible residual proximal networks from
`checkpoints_scratch5/model_best.pth` with strict state-dict loading. The current
run trains and validates for 25 epochs at the requested 50 ADMM iterations with
a batch size of 3.

The restored proximal has one signal input channel, spectral normalization on
all convolutions, the original Softplus MLP gate, and the residual update
`x + gate * residual`. Its gate receives the raw iteration index and update
RMS, matching the pretrained checkpoint. The solver likewise uses the original
unit-penalty ADMM forward semantics. Current deep supervision, convergence
metrics, gradient clipping, EMA selection, deterministic validation, and
resumable training state are retained.

CUDA training and validation use FP16 autocast with dynamic gradient scaling.
Learned activations, recurrent ADMM state, and saved iterates use FP16; FFT
inputs and physics calculations are explicitly promoted to FP32 because
half-precision FFT support is shape- and hardware-dependent. The scaler is
saved in `training_last.pth`, so mixed-precision resumes are exact.

The recovery run disables global unit scaling, susceptibility-derived weight
textures, and sparse noise outliers because those distributions were not used
to pretrain scratch5. Current complex Gaussian phase-noise simulation remains
active.

Checkpoints are written atomically under `checkpoints_scratch5_k50/`:

- `model_best.pth`: best EMA weights at the active stage;
- `model_best_k{depth}.pth`: best EMA weights for each depth;
- `model_last.pth`: latest validated EMA weights;
- `training_last.pth`: resumable optimizer, scheduler, EMA, RNG, and loader
  state; and
- `solver_config.pth`: architecture and solver semantics.

Set `RESUME_CKPT` to `training_last.pth` to resume this version exactly. Leave
`RESUME_CKPT = None` to begin a new K=50 adaptation from scratch5.
