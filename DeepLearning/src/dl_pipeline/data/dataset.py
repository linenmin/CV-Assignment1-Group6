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


class TripletFaceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform=None,
        lookalike_csv_path: str | Path | None = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

        self.targets_frame = self.frame[self.frame["class"].isin([1, 2])].reset_index(drop=True)
        self.other_frame = self.frame[self.frame["class"] == 0]

        if lookalike_csv_path and Path(lookalike_csv_path).exists():
            lookalike_df = pd.read_csv(lookalike_csv_path)
            michael_like_ids = lookalike_df[lookalike_df["cluster"] == 1]["id"].tolist()
            sarah_like_ids = lookalike_df[lookalike_df["cluster"] == 0]["id"].tolist()
        else:
            michael_like_ids = []
            sarah_like_ids = []

        self.jesse_neg_frame = self.frame[self.frame["id"].isin(michael_like_ids)]
        self.mila_neg_frame = self.frame[self.frame["id"].isin(sarah_like_ids)]

    def __len__(self) -> int:
        return len(self.targets_frame)

    def _load_image(self, row: pd.Series):
        image = cv2.imread(str(Path(row["image_path"])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __getitem__(self, index: int):
        import random

        anchor_row = self.targets_frame.iloc[index]
        target_class = int(anchor_row["class"])

        anchor_image = self._load_image(anchor_row)

        pos_frame = self.targets_frame[self.targets_frame["class"] == target_class]
        pos_row = pos_frame.sample(n=1).iloc[0]
        pos_image = self._load_image(pos_row)

        if target_class == 1:
            neg_frame = self.jesse_neg_frame if not self.jesse_neg_frame.empty else self.other_frame
        else:
            neg_frame = self.mila_neg_frame if not self.mila_neg_frame.empty else self.other_frame

        if neg_frame.empty:
            neg_frame = self.targets_frame[self.targets_frame["class"] != target_class]

        neg_row = neg_frame.sample(n=1).iloc[0]
        neg_image = self._load_image(neg_row)

        return anchor_image, pos_image, neg_image, target_class

