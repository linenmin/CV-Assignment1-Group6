from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dl_pipeline.common.paths import ensure_dir


def append_registry_row(registry_path: Path, row: dict[str, Any]) -> None:
    ensure_dir(registry_path.parent)
    payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    if registry_path.exists():
        frame = pd.read_csv(registry_path)
        frame = pd.concat([frame, pd.DataFrame([payload])], ignore_index=True)
    else:
        frame = pd.DataFrame([payload])
    frame.to_csv(registry_path, index=False)
