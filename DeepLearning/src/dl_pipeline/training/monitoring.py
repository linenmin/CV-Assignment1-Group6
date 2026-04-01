from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorConfig:
    metric: str
    mode: str


def resolve_monitor_config(train_config: dict) -> MonitorConfig:
    metric = train_config.get("monitor_metric", "val_acc")
    mode = train_config.get("monitor_mode")

    if mode is None:
        mode = "min" if metric == "val_loss" else "max"

    return MonitorConfig(metric=metric, mode=mode)
