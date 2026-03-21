from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(image_size: int):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=12, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: int):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
