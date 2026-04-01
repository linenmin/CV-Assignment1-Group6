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
        use_horizontal_flip: bool = True,
        use_affine: bool = False,
        use_degradation_pack: bool = False,
        normalization: str = "imagenet",
    ) -> None:
        super().__init__()
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_horizontal_flip = use_horizontal_flip
        self.use_affine = use_affine
        self.use_degradation_pack = use_degradation_pack
        self.normalization = normalization

    def setup(self, stage: str | None = None) -> None:
        self.train_df = pd.read_csv(self.train_csv)
        self.val_df = pd.read_csv(self.val_csv)
        self.test_df = pd.read_csv(self.test_csv)

        self.train_dataset = FaceDataset(
            self.train_df,
            build_train_transform(
                self.image_size,
                use_horizontal_flip=self.use_horizontal_flip,
                use_affine=self.use_affine,
                use_degradation_pack=self.use_degradation_pack,
                normalization=self.normalization,
            ),
            True,
        )
        self.val_dataset = FaceDataset(
            self.val_df,
            build_eval_transform(self.image_size, normalization=self.normalization),
            True,
        )
        self.test_dataset = FaceDataset(
            self.test_df,
            build_eval_transform(self.image_size, normalization=self.normalization),
            False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
