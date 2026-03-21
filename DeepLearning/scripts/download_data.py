from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.paths import ensure_dir, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Kaggle 比赛数据。")
    parser.add_argument("--competition", required=True)
    parser.add_argument("--output-dir", default="data/raw/kul-computer-vision-ga-1-2026")
    args = parser.parse_args()

    output_dir = ensure_dir(project_path(args.output_dir))
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError("未找到 ~/.kaggle/kaggle.json，无法下载 Kaggle 数据。")

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", args.competition, "-p", str(output_dir)],
        check=True,
    )
    for zip_path in output_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(output_dir)
    print(f"数据已下载并解压到: {output_dir}")


if __name__ == "__main__":
    main()
