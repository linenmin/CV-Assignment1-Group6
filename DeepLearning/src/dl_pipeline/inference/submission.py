from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_submission_dataframe(test_df: pd.DataFrame, predictions: list[int]) -> pd.DataFrame:
    submission = pd.DataFrame({"id": test_df["id"].tolist(), "class": [int(item) for item in predictions]})
    submission = submission.set_index("id")
    submission.index.name = "id"
    return submission


def save_submission_dataframe(submission: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path)
    return output_path
