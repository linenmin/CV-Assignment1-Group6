from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.preprocess.build_multiface_selected_dataset import build_multiface_selected_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="构建多脸自动选中的训练/验证/测试输入。")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    artifacts = build_multiface_selected_dataset(config)
    print(
        "多脸选脸数据已生成。"
        f" train={len(artifacts.train_df)} val={len(artifacts.val_df)} test={len(artifacts.test_df)}"
        f" theta_jesse={artifacts.theta_jesse:.6f} theta_mila={artifacts.theta_mila:.6f}"
    )


if __name__ == "__main__":
    main()
