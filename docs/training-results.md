# Joint Training Results

## Run status

Training completed all 50 epochs on 9 September 2026. The run performed 5,849 optimizer updates in approximately 35 h 43 min and peaked at 51.11 GiB of allocated GPU memory. Both susceptibility (chi) and harmonic-component networks were optimized. One nonfinite update was skipped at the start of epoch 1; all later epochs completed without skipped updates.

The selected checkpoint is `artifacts/checkpoints/checkpoints_joint_20260908_001534/model_best.pth`. It contains EMA weights from epoch 50, chosen by subject-mean chi NRMSE at 55 iterations.

## Final validation results

Validation used 16 held-out source subjects, three deterministic synthetic realizations per subject, and macro-averaging across subjects.

| Unrolled iterations | Chi NRMSE | Harmonic NRMSE |
|---:|---:|---:|
| 20 | 0.61046 | 0.02491 |
| 30 | 0.56986 | 0.01949 |
| 55 | **0.53296** | **0.01860** |

From epoch 1 to epoch 50, the 55-iteration chi error fell from 3.18418 to 0.53296 (83.3%), while harmonic error fell from 0.07359 to 0.01860 (74.7%). Chi continued improving through the last epoch, from 0.54141 at epoch 45 to 0.53296.

![Validation convergence and comparison](../artifacts/checkpoints/checkpoints_joint_20260908_001534/final_results_comparison.png)

## Comparison with scratch7

`artifacts/checkpoints/checkpoints_scratch7/model_best.pth` was evaluated on exactly the same 48 deterministic validation realizations using its original legacy ADMM implementation.

| Iterations | New chi | Scratch7 chi | New vs scratch7 | New harmonic | Scratch7 harmonic | New vs scratch7 |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.61046 | **0.56972** | +7.2% | **0.02491** | 0.05750 | -56.7% |
| 30 | 0.56986 | **0.51772** | +10.1% | **0.01949** | 0.03065 | -36.4% |
| 55 | 0.53296 | **0.48259** | +10.4% | 0.01860 | **0.01529** | +21.7% |

Lower is better. Scratch7 gives the lowest final error at 55 iterations for both outputs. The new model recovers the harmonic component much faster at 20 and 30 iterations, but this advantage disappears by iteration 55. Its chi result remains worse at every measured depth.

The scratch7 harmonic branch was effectively inherited from scratch6 rather than trained during the scratch7 run: its parameter relative L2 change is 4.93e-8, compared with 1.14e-2 for the chi branch. The new run actively trained both branches.

## Interpretation

This benchmark is useful for selecting a current checkpoint, but it is not a clean generalization comparison for scratch7. Scratch7 used a random volume-level split, so augmented volumes from the same underlying subjects could occur in both training and validation. The new run split by source subject before augmentation; however, scratch7 had already trained on subjects included in this new validation set. The comparison therefore favors scratch7.

For minimum error on this synthetic benchmark at 55 iterations, scratch7 is presently the stronger checkpoint. The new model provides corrected projected physics, subject-safe training, trained harmonic weights, and substantially better early-iteration harmonic estimates. A definitive model comparison requires a separate subject-level test set unseen by both checkpoints.
