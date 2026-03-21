from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def make_stratified_holdout_split(
    frame: pd.DataFrame,
    val_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        frame,
        test_size=val_ratio,
        random_state=random_state,
        stratify=frame["class"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
