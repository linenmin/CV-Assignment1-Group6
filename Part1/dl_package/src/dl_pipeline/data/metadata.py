from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_metadata(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train_df = pd.read_csv(raw_dir / "train_set.csv", index_col=0)
    test_df = pd.read_csv(raw_dir / "test_set.csv", index_col=0)

    train_df = train_df.reset_index().rename(columns={"index": "id"})
    test_df = test_df.reset_index().rename(columns={"index": "id"})

    train_df["source_path"] = train_df["id"].apply(lambda idx: str(raw_dir / "train" / f"train_{idx}.npy"))
    test_df["source_path"] = test_df["id"].apply(lambda idx: str(raw_dir / "test" / f"test_{idx}.npy"))
    return train_df, test_df
