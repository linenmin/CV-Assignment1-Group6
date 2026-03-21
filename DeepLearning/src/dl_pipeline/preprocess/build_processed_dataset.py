from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dl_pipeline.common.paths import ensure_dir
from dl_pipeline.data.metadata import load_raw_metadata
from dl_pipeline.face_detection.haar_detector import HaarFaceDetector


def _build_detector(detector_name: str, cache_dir: str | Path, face_size: int):
    if detector_name != "haar":
        raise ValueError(f"当前第一版只实现了 haar 检测器，收到: {detector_name}")
    return HaarFaceDetector(cache_dir=cache_dir, face_size=face_size)


def _load_competition_image(npy_path: str | Path) -> np.ndarray:
    image = np.load(npy_path, allow_pickle=False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_processed_record(row: dict[str, object], output_path: str | Path) -> dict[str, object]:
    record = {
        "id": int(row["id"]),
        "image_path": str(output_path),
        "source_path": str(row["source_path"]),
    }
    if "class" in row and row["class"] is not None:
        record["class"] = int(row["class"])
    return record


def _process_frame(frame: pd.DataFrame, output_dir: Path, detector) -> pd.DataFrame:
    ensure_dir(output_dir)
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        image_rgb = _load_competition_image(row["source_path"])
        face_rgb = detector.extract_primary_face(image_rgb)
        output_path = output_dir / f"{row['id']}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
        records.append(build_processed_record(row=row, output_path=output_path))
    return pd.DataFrame(records)


def build_processed_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    detector_name: str,
    face_size: int,
    detector_cache_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    ensure_dir(processed_dir)

    train_df, test_df = load_raw_metadata(raw_dir)
    detector = _build_detector(detector_name, detector_cache_dir, face_size)
    processed_train = _process_frame(train_df, processed_dir / "train", detector)
    processed_test = _process_frame(test_df, processed_dir / "test", detector)

    processed_train.to_csv(processed_dir / "train_metadata.csv", index=False)
    processed_test.to_csv(processed_dir / "test_metadata.csv", index=False)
    return processed_train, processed_test
