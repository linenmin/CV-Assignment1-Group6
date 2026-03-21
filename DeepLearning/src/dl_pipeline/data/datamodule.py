from __future__ import annotations

from pathlib import Path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from dl_pipeline.data.dataset import FaceDataset
from dl_pipeline.data.transforms import build_eval_transform, build_train_transform


class FaceDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv: str | Path,
        val_csv: str | Path,
        test_csv: str | Path,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        self.train_df = pd.read_csv(self.train_csv)
        self.val_df = pd.read_csv(self.val_csv)
        self.test_df = pd.read_csv(self.test_csv)

        self.train_dataset = FaceDataset(self.train_df, build_train_transform(self.image_size), True)
        self.val_dataset = FaceDataset(self.val_df, build_eval_transform(self.image_size), True)
        self.test_dataset = FaceDataset(self.test_df, build_eval_transform(self.image_size), False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
