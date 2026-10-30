"""Audit actual training-loader susceptibility and synthesized field scales.

Run from the repository root with ``python -m scripts.report_augmented_training``.
The report uses epoch zero and fixed seeds, without starting a training job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.report_synthetic_data import (
    Distribution,
    EDGES,
    load_external_distribution,
)
from whnet.adaptive_wh_tv_validation import (
    load_external_cases,
    physical_external_cases,
)
from whnet.data import FieldSynthesizer, TrainingVolumes, training_records
from whnet.paths import PROJECT_ROOT, RUNS_DIR, TRAINING_CACHE_DIR, TRAINING_DATA_DIR
from whnet.training import BalancedSourceSampler, make_subject_split

OUTPUT = PROJECT_ROOT / "docs" / "figures" / "training_distribution"
REPORT = PROJECT_ROOT / "docs" / "training-distribution-report.md"
OBSERVED_RUN = RUNS_DIR / "joint_20261002_214544"
PREVIOUS_RUN = RUNS_DIR / "joint_20260929_202153"


def masked_rms(volume: torch.Tensor, mask: torch.Tensor) -> float:
    return float(((volume.square() * mask).sum() / mask.sum()).sqrt())


def select_examples(indices: list[int], dataset: TrainingVolumes) -> list[int]:
    groups = {"original": [], "synthetic": []}
    for index in indices:
        record = dataset.records[index % len(dataset.records)]
        group = (
            "synthetic" if record["subject"].startswith("synthetic:") else "original"
        )
        groups[group].append(index)
    return [
        group[int(position)]
        for group in groups.values()
        if group
        for position in np.linspace(0, len(group) - 1, min(8, len(group)))
    ]


def scan(name: str, dataset: TrainingVolumes, indices: list[int], mode: str):
    distribution = Distribution()
    source = {"original": Distribution(), "synthetic": Distribution()}
    volume_rms = []
    volume_histogram = np.zeros(len(EDGES) - 1, dtype=np.float64)
    examples = set(select_examples(indices, dataset))
    field_rows = []
    synthesizer = FieldSynthesizer((160, 160, 160), "cpu")
    for position, index in enumerate(indices, start=1):
        sample = dataset[index]
        chi = sample["chi"].numpy().squeeze(0)
        mask = sample["mask"].numpy().squeeze(0)
        previous = distribution.histogram.copy()
        distribution.add(chi, mask)
        volume_histogram += (distribution.histogram - previous) / mask.sum()
        cohort = (
            "synthetic" if sample["subject"].startswith("synthetic:") else "original"
        )
        source[cohort].add(chi, mask)
        volume_rms.append(float(np.sqrt(np.mean(np.square(chi[mask])))))
        if index in examples:
            batch = {
                key: value.unsqueeze(0)
                if isinstance(value, torch.Tensor)
                else torch.tensor([value])
                for key, value in sample.items()
                if key in ("chi", "mask", "dilated", "seed")
            }
            *_, diagnostics = synthesizer(
                batch, background_mode=mode, return_diagnostics=True
            )
            field_rows.append(
                {
                    "source": cohort,
                    **{key: float(value[0]) for key, value in diagnostics.items()},
                }
            )
        if position % 100 == 0:
            print(f"{name}: {position}/{len(indices)}", flush=True)
    summary = distribution.summary()
    summary["volume_rms_p10_p50_p90_ppm"] = np.quantile(
        volume_rms, [0.1, 0.5, 0.9]
    ).tolist()
    cumulative = np.cumsum(volume_histogram) / distribution.volumes
    summary["volume_weighted_chi_p01_p50_p99_ppm"] = [
        float(EDGES[min(np.searchsorted(cumulative, quantile), len(EDGES) - 2)])
        for quantile in (0.01, 0.5, 0.99)
    ]
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    summary["volume_weighted_abs_above_005_fraction"] = float(
        volume_histogram[np.abs(centers) > 0.05].sum() / distribution.volumes
    )
    summary["source_counts"] = {key: value.volumes for key, value in source.items()}
    summary["source_summaries"] = {
        key: value.summary() for key, value in source.items() if value.volumes
    }
    return distribution, summary, field_rows


def reference_fields(cases):
    result = {}
    for case in cases:
        shape = case.chi.shape[-3:]
        local = torch.fft.irfftn(
            torch.fft.rfftn(case.chi, dim=(-3, -2, -1))
            * case.dipole[..., : shape[-1] // 2 + 1],
            s=shape,
            dim=(-3, -2, -1),
        )
        residual = (
            case.background if case.background is not None else case.phase - local
        )
        result[case.name] = {
            "chi_rms_ppm": masked_rms(case.chi, case.mask),
            "local_rms_ppm": masked_rms(local, case.mask),
            "phase_rms_ppm": masked_rms(case.phase, case.mask),
            "background_or_residual_rms_ppm": masked_rms(residual, case.mask),
            "local_to_background_or_residual": masked_rms(local, case.mask)
            / masked_rms(residual, case.mask),
            "residual_is_reference_background": case.background is not None,
        }
    return result


def observed_run():
    path = OBSERVED_RUN / "metrics.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    updates = [row for row in rows if row.get("event") == "train"]
    validation = [row for row in rows if row.get("event") == "validation"]
    if not updates or not validation:
        return None
    previous_path = PREVIOUS_RUN / "metrics.jsonl"
    previous_updates = (
        [
            json.loads(line)
            for line in previous_path.read_text().splitlines()
            if '"event": "train"' in line
        ]
        if previous_path.is_file()
        else []
    )
    return {
        "run": str(OBSERVED_RUN.relative_to(PROJECT_ROOT)),
        "updates": len(updates),
        "clipped_both_fraction": sum(
            row["clipped_chi"] and row["clipped_phi"] for row in updates
        )
        / len(updates),
        "last_lr": updates[-1]["lr_chi"],
        "previous_run": (
            {
                "last_lr": previous_updates[-1]["lr_chi"],
                "clipped_both_fraction": sum(
                    row["clipped_chi"] and row["clipped_phi"]
                    for row in previous_updates
                )
                / len(previous_updates),
            }
            if previous_updates
            else None
        ),
        "validation": [
            {
                "epoch": row["epoch"],
                "real_chi_nrmse": row["val_chi_k60"],
                "cosmos_chi_nrmse": row["external_cosmos_chi_nrmse"],
                "sim2_chi_nrmse": row["external_sim2_chi_nrmse"],
            }
            for row in validation
        ],
    }


def draw(distributions, field_rows, references):
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    fig, axis = plt.subplots(figsize=(8, 4.8))
    colors = {
        "Current loader": "#8c4778",
        "Revised loader": "#1c77a6",
        "COSMOS": "#18865f",
        "sim2": "#dc6844",
    }
    for name, distribution in distributions.items():
        density = distribution.histogram / (distribution.voxels * (EDGES[1] - EDGES[0]))
        axis.plot(
            centers,
            np.convolve(density, np.ones(21) / 21, "same"),
            label=name,
            color=colors[name],
        )
    axis.set(
        xlim=(-0.22, 0.22),
        xlabel="χ (ppm)",
        ylabel="Masked-voxel density",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "augmented_chi.png", dpi=160)
    plt.close(fig)

    draw_field_ratios(field_rows, references)


def draw_field_ratios(field_rows, references):
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for position, (name, rows) in enumerate(field_rows.items()):
        values = [row["local_to_background"] for row in rows]
        axis.scatter(
            np.full(len(values), position) + np.linspace(-0.15, 0.15, len(values)),
            values,
            alpha=0.75,
            label=name,
        )
    for name, value in references.items():
        axis.axhline(
            value["local_to_background_or_residual"],
            label=f"{name} reference",
            linestyle="--",
            color="#18865f" if name == "cosmos" else "#dc6844",
        )
    axis.set(
        xticks=[0, 1],
        xticklabels=list(field_rows),
        yscale="log",
        ylabel="Local / background field RMS",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "field_ratios.png", dpi=160)
    plt.close(fig)


def write_report(summary):
    distributions = summary["distributions"]
    references = summary["reference_fields"]
    rows = []
    for name in ("Current loader", "Revised loader", "COSMOS", "sim2"):
        entry = distributions[name]
        rms = entry.get("volume_rms_p10_p50_p90_ppm")
        rms_label = f"{rms[1]:.4f}" if rms else f"{entry['chi_std_ppm']:.4f}"
        rows.append(
            f"| {name} | {entry['volumes']} | {rms_label} | "
            f"{entry['chi_p01_ppm']:.4f} / {entry['chi_p99_ppm']:.4f} | "
            f"{entry['abs_above_005_fraction']:.1%} | "
            f"{entry['abs_above_010_fraction']:.1%} |"
        )
    current = distributions["Current loader"]
    revised = distributions["Revised loader"]
    current_volume_range = current["volume_weighted_chi_p01_p50_p99_ppm"]
    revised_volume_range = revised["volume_weighted_chi_p01_p50_p99_ppm"]
    observed = summary["observed_run"]
    observed_text = ""
    if observed:
        first, last = observed["validation"][0], observed["validation"][-1]
        previous_text = ""
        if previous := observed["previous_run"]:
            previous_text = (
                f" The earlier run also clipped both branches on "
                f"{previous['clipped_both_fraction']:.0%} of updates, so "
                "clipping alone does not explain the decline. Its final "
                f"learning rate was {previous['last_lr']:.1e}; the declining "
                f"run reached {observed['last_lr']:.1e}."
            )
        observed_text = (
            f"The paused run `{observed['run']}` clipped both branches on "
            f"{observed['clipped_both_fraction']:.0%} of its "
            f"{observed['updates']} logged updates. From epoch "
            f"{first['epoch']} to {last['epoch']}, real-subject χ NRMSE changed "
            f"from {first['real_chi_nrmse']:.3f} to "
            f"{last['real_chi_nrmse']:.3f}, COSMOS from "
            f"{first['cosmos_chi_nrmse']:.3f} to "
            f"{last['cosmos_chi_nrmse']:.3f}, and sim2 from "
            f"{first['sim2_chi_nrmse']:.3f} to "
            f"{last['sim2_chi_nrmse']:.3f}. This does not isolate the data "
            "change from learning-rate or optimization effects."
            f"{previous_text}\n"
        )
    report = f"""# Training susceptibility and field-scale audit

**Method.** Epoch-zero samples from the actual subject split and dataloader, seed 0.
The current path uses every training record once with χ multiplied by U(0.5, 3).
The revised path draws {revised["source_counts"]["original"]} original and {revised["source_counts"]["synthetic"]} synthetic samples using the balanced sampler.
It draws each volume's masked χ RMS from 0.018–0.040 ppm (95%) or
0.040–0.080 ppm (5%). Spatial flips and permutations are
applied by `TrainingVolumes` in both paths. χ density and tails weight every masked
voxel equally; the RMS column is the median across sampled volumes for training
and the single-volume masked RMS for the references. COSMOS and sim2 are in
physical ppm. Field ratios use eight deterministic examples per training source
and mode; they illustrate coverage rather than an exhaustive field distribution.
Representative mask/χ slices are in the [generated data report](synthetic-data-report.md).

| Cohort | Samples | Median volume χ RMS (ppm) | χ p1 / p99 (ppm) | \\|χ\\| > 0.05 | \\|χ\\| > 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

![Augmented susceptibility distributions](figures/training_distribution/augmented_chi.png)

The existing loader's combined masked χ standard deviation is
**{current["chi_std_ppm"]:.4f} ppm**; the revised mix is
**{revised["chi_std_ppm"]:.4f} ppm**. The revised sampler retains a broad tail while
placing more volume RMS values near the two references, both close to 0.028 ppm.
With each training volume weighted equally, χ p1/p99 changes from
{current_volume_range[0]:.4f}/{current_volume_range[2]:.4f} ppm to
{revised_volume_range[0]:.4f}/{revised_volume_range[2]:.4f} ppm.
The original and synthetic source summaries, including per-volume RMS quantiles,
are in [augmented_statistics.json](figures/training_distribution/augmented_statistics.json).
sim2 contains a rare negative extreme near -2.03 ppm; these central-range
sampling changes do not intentionally replicate that isolated tail. Its spatial
support should be checked before adding such extreme χ to training.
The revised loader has {revised["positive_fraction"]:.1%} positive masked voxels,
between COSMOS ({distributions["COSMOS"]["positive_fraction"]:.1%}) and sim2 ({distributions["sim2"]["positive_fraction"]:.1%}). Amplitude scaling does not
change this sign balance or replace their anatomical spatial patterns.
COSMOS's positive p99 ({distributions["COSMOS"]["chi_p99_ppm"]:.4f} ppm) remains
above the revised loader's ({revised["chi_p99_ppm"]:.4f} ppm), so upper-tail
coverage is approximate.

## Field scales

| Reference | χ RMS | Local field RMS | Total phase RMS | Background or residual RMS | Local / background or residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| COSMOS | {references["cosmos"]["chi_rms_ppm"]:.4f} | {references["cosmos"]["local_rms_ppm"]:.4f} | {references["cosmos"]["phase_rms_ppm"]:.4f} | {references["cosmos"]["background_or_residual_rms_ppm"]:.4f} | {references["cosmos"]["local_to_background_or_residual"]:.3f} |
| sim2 | {references["sim2"]["chi_rms_ppm"]:.4f} | {references["sim2"]["local_rms_ppm"]:.4f} | {references["sim2"]["phase_rms_ppm"]:.4f} | {references["sim2"]["background_or_residual_rms_ppm"]:.4f} | {references["sim2"]["local_to_background_or_residual"]:.3f} |

For sim2, the residual is phase minus the dipole forward field from its χ
reference. It includes model mismatch and noise and is **not** a measured
background field. COSMOS uses its saved background reference.

![Sampled field ratios](figures/training_distribution/field_ratios.png)

The new synthesis draws a local/background RMS ratio log-uniformly from
0.03–3, spanning these two cases. A stronger phase normalization alone cannot
remove differences in the relative local and background fields. The network
now uses a 1e-6 input-scale floor, and the high-χ loss is normalized by each
target's masked RMS. Positive scaling of the full field should therefore be
approximately equivariant above the numerical floor. Changing only χ while
holding background fixed changes the inverse problem and must be handled by
the field-ratio sampling.

## Training decision

{observed_text}
Treat COSMOS and sim2 as development validation cases, never training voxels.
Compare same-initialization 60-iteration ablations for current sampling,
χ balancing, χ plus field-ratio balancing, and normalized high-χ loss.
Select using the worse of the two external χ NRMSE values relative to the
same 60-iteration baseline, while reporting real-subject validation and
gradient clipping. The current 60-iteration run worsened on the held-out
subjects as well as the external cases after epoch 2, so data distribution
alone is not yet a proven cause; optimization stability remains a competing
explanation. A full training run has not been launched by this audit.
"""
    REPORT.write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=TRAINING_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=TRAINING_CACHE_DIR)
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, validation = make_subject_split(training_records(args.data_dir), 0.1, 0)
    assert all(not row["subject"].startswith("synthetic:") for row in validation)
    current_set = TrainingVolumes(train, args.cache_dir, seed=0)
    revised_set = TrainingVolumes(
        train, args.cache_dir, seed=0, chi_scale_mode="rms_mixture"
    )
    current, current_summary, current_fields = scan(
        "Current loader", current_set, list(range(len(current_set))), "legacy"
    )
    revised, revised_summary, revised_fields = scan(
        "Revised loader",
        revised_set,
        list(BalancedSourceSampler(train, 0.25, seed=0)),
        "ratio",
    )
    cases = physical_external_cases(load_external_cases(sim2_factor=1.0))
    references = reference_fields(cases)
    distributions = {
        "Current loader": current,
        "Revised loader": revised,
        **{
            case.name if case.name == "sim2" else "COSMOS": load_external_distribution(
                case
            )
            for case in cases
        },
    }
    summary = {
        "method": {
            "seed": 0,
            "epoch": 0,
            "train_records": len(train),
            "validation_subjects": len({row["subject"] for row in validation}),
            "field_examples_per_source_per_mode": 8,
            "revised_chi_rms_core_ppm": [0.018, 0.04],
            "revised_chi_rms_tail_ppm": [0.04, 0.08],
            "revised_chi_rms_tail_probability": 0.05,
        },
        "distributions": {
            "Current loader": current_summary,
            "Revised loader": revised_summary,
            **{
                name: dist.summary()
                for name, dist in distributions.items()
                if name in ("COSMOS", "sim2")
            },
        },
        "reference_fields": references,
        "observed_run": observed_run(),
        "training_field_examples": {
            "Current loader": current_fields,
            "Revised loader": revised_fields,
        },
    }
    (OUTPUT / "augmented_statistics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    draw(distributions, summary["training_field_examples"], references)
    write_report(summary)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
