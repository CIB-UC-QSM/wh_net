"""Compare 60-iteration QSM ablations using their saved best-validation weights.

Example: python -m scripts.compare_scale_runs BASELINE_RUN CANDIDATE_RUN ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def best_checkpoint_metrics(directory: Path):
    config = json.loads((directory / "config.json").read_text())
    split = json.loads((directory / "split.json").read_text())
    iterations = config["final_iters"]
    validation_key = f"val_chi_k{iterations}"
    rows = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
    ]
    available = [
        row
        for row in rows
        if row.get("event") == "validation"
        and validation_key in row
        and "external_cosmos_chi_nrmse" in row
        and "external_sim2_chi_nrmse" in row
    ]
    if not available or not (directory / "model_best.pth").is_file():
        raise ValueError(f"{directory} lacks a scored best checkpoint")
    best = min(available, key=lambda row: row[validation_key])
    if best[validation_key] != min(
        row[validation_key]
        for row in rows
        if row.get("event") == "validation" and validation_key in row
    ):
        raise ValueError(f"{directory} best checkpoint has no external score")
    return config, split, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    baseline_config, baseline_split, baseline = best_checkpoint_metrics(args.baseline)
    if baseline_config["final_iters"] != 60:
        raise ValueError("Scale comparison requires a 60-iteration baseline")
    baseline_subjects = {row["subject"] for row in baseline_split["validation"]}
    key = "val_chi_k60"
    records = []
    for directory in [args.baseline, *args.candidates]:
        config, split, row = best_checkpoint_metrics(directory)
        if config["final_iters"] != 60:
            raise ValueError(f"{directory} uses a different iteration count")
        if {entry["subject"] for entry in split["validation"]} != baseline_subjects:
            raise ValueError(f"{directory} uses a different subject validation split")
        for setting in ("seed", "init_ckpt", "peak_lr"):
            if config[setting] != baseline_config[setting]:
                raise ValueError(f"{directory} differs in {setting}")
        cosmos = row["external_cosmos_chi_nrmse"]
        sim2 = row["external_sim2_chi_nrmse"]
        worst_ratio = max(
            cosmos / baseline["external_cosmos_chi_nrmse"],
            sim2 / baseline["external_sim2_chi_nrmse"],
        )
        eligible = row[key] <= 1.05 * baseline[key]
        records.append((directory, row, worst_ratio, eligible))
    print(
        "run | epoch | real χ NRMSE | COSMOS χ NRMSE | sim2 χ NRMSE | worst ratio | guardrail"
    )
    print("--- | ---: | ---: | ---: | ---: | ---: | ---")
    for directory, row, ratio, eligible in records:
        print(
            f"{directory} | {row['epoch']} | {row[key]:.4f} | "
            f"{row['external_cosmos_chi_nrmse']:.4f} | "
            f"{row['external_sim2_chi_nrmse']:.4f} | {ratio:.3f} | "
            f"{'pass' if eligible else 'fail'}"
        )
    candidates = [record for record in records[1:] if record[3]]
    if candidates:
        selected = min(candidates, key=lambda record: record[2])
        print(f"Selected candidate: {selected[0]}/model_best.pth")
    else:
        print("No candidate passes the real-subject guardrail")


if __name__ == "__main__":
    main()
