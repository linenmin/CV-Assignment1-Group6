from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from dl_pipeline.common.paths import PROJECT_ROOT


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_experiment_config(experiment_config_path: str | Path) -> dict[str, Any]:
    base_config = load_yaml("configs/base.yaml")
    experiment_config = load_yaml(experiment_config_path)
    merged = deep_merge_dicts(base_config, experiment_config)
    merged["config_path"] = str(Path(experiment_config_path))
    return merged
