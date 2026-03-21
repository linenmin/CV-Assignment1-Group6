from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def summarize_classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(frame["target"], frame["prediction"])),
        "macro_f1": float(f1_score(frame["target"], frame["prediction"], average="macro")),
    }
