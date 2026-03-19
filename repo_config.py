from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DOCS_DIR = REPO_ROOT / "docs"
EXAMPLES_DIR = REPO_ROOT / "examples"
ANALYSIS_DIR = REPO_ROOT / "analysis"

DATA_ROOT_ENV = "OPENSIM_MUSCLE_NN_DATA_ROOT"

GENERIC_MALE_REL = Path(
    "generic/Thoracolumbar Model/Male_Thoracolumbar_Spine_V1/Thoracolumbar_Spine_With_RibCage.osim"
)
GENERIC_FEMALE_REL = Path(
    "generic/Thoracolumbar Model/Female_Thoracolumbar_Spine_V1/Female_Thoracolumbar_Spine_Model.osim"
)


def get_data_root() -> Path | None:
    raw = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def require_data_root() -> Path:
    data_root = get_data_root()
    if data_root is None:
        raise RuntimeError(
            f"Set {DATA_ROOT_ENV} to the root directory containing the Male/, Female/, and generic/ folders."
        )
    return data_root


def male_template_path(data_root: Path | None = None) -> Path:
    root = data_root or require_data_root()
    return root / GENERIC_MALE_REL


def female_template_path(data_root: Path | None = None) -> Path:
    root = data_root or require_data_root()
    return root / GENERIC_FEMALE_REL
