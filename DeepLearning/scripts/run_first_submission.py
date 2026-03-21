from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_step(command: list[str]) -> None:
    print(f">>> {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="顺序执行第一版 submission 流程。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--competition", default="kul-computer-vision-ga-1-2026")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    if not args.skip_download:
        _run_step(
            [
                python_exe,
                str(scripts_dir / "download_data.py"),
                "--competition",
                args.competition,
            ]
        )

    for script_name in ["prepare_faces.py", "make_split.py", "train.py", "predict.py"]:
        _run_step([python_exe, str(scripts_dir / script_name), "--config", args.config])


if __name__ == "__main__":
    main()
