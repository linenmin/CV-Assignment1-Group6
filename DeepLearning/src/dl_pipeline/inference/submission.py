from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_submission_dataframe(test_df: pd.DataFrame, predictions: list[int]) -> pd.DataFrame:
    submission = pd.DataFrame({"id": test_df["id"].tolist(), "class": [int(item) for item in predictions]})
    submission = submission.set_index("id")
    submission.index.name = "id"
    return submission


def apply_submission_postprocess(
    submission: pd.DataFrame,
    postprocess_config: dict | None,
) -> pd.DataFrame:
    if not postprocess_config or not bool(postprocess_config.get("enabled", False)):
        return submission

    mode = str(postprocess_config.get("mode", "")).strip().lower()
    processed = submission.copy()

    if mode == "override_labels":
        id_to_class = postprocess_config.get("id_to_class", {})
        if not isinstance(id_to_class, dict):
            raise ValueError("submission_postprocess.id_to_class 必须是 dict。")
        for sample_id, target_class in id_to_class.items():
            sample_id_int = int(sample_id)
            if sample_id_int not in processed.index:
                continue
            processed.loc[sample_id_int, "class"] = int(target_class)
        return processed

    raise ValueError(f"未知的 submission_postprocess mode: {mode}")


def save_submission_dataframe(submission: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path)
    return output_path
