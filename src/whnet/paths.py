"""Canonical repository paths used by command-line tools and experiments."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JULIA_ENV_DIR = PROJECT_ROOT / "experiments" / "julia"
DATA_ROOT = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_ROOT / "training" / "joint"
LEGACY_TRAINING_DATA_DIR = DATA_ROOT / "training" / "legacy"
SOURCE_DATA_DIR = DATA_ROOT / "sources"
EVALUATION_DATA_DIR = DATA_ROOT / "evaluation"
EVALUATION_VOLUMES_DIR = EVALUATION_DATA_DIR / "volumes"

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_ROOT / "checkpoints"
RUNS_DIR = ARTIFACTS_ROOT / "runs"
EXPORTS_DIR = ARTIFACTS_ROOT / "exports"
TRAINING_CACHE_DIR = ARTIFACTS_ROOT / "cache" / "training"
