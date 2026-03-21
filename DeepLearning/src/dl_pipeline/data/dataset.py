from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None, has_targets: bool = True):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.has_targets = has_targets

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = cv2.imread(str(Path(row["image_path"])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.has_targets:
            return image, int(row["class"])
        return image, int(row["id"])
