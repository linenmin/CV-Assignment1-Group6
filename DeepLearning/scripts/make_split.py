from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.data.split import make_stratified_holdout_split


def main() -> None:
    parser = argparse.ArgumentParser(description="生成固定训练/验证划分。")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    processed_dir = project_path(config["data"]["processed_dir"])
    splits_dir = ensure_dir(project_path(config["data"]["splits_dir"]))

    train_metadata = pd.read_csv(processed_dir / "train_metadata.csv")
    test_metadata = pd.read_csv(processed_dir / "test_metadata.csv")
    train_df, val_df = make_stratified_holdout_split(
        train_metadata,
        val_ratio=config["data"]["val_ratio"],
        random_state=config["seed"],
    )

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_metadata.to_csv(splits_dir / "test.csv", index=False)
    print(f"划分文件已保存到: {splits_dir}")


if __name__ == "__main__":
    main()
