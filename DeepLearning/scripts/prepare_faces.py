from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.preprocess.build_processed_dataset import build_processed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="裁脸并生成处理后图像。")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    build_processed_dataset(
        raw_dir=config["data"]["raw_dir"],
        processed_dir=config["data"]["processed_dir"],
        detector_name=config["data"]["detector"],
        face_size=config["data"]["face_size"],
        detector_cache_dir=config["data"]["detector_cache_dir"],
    )
    print("处理后数据已生成。")


if __name__ == "__main__":
    main()
