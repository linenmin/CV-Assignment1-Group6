from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _build_normalize(normalization: str):
    if normalization == "face":
        return A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return A.Normalize()


def build_train_transform(
    image_size: int,
    use_horizontal_flip: bool = True,
    use_affine: bool = False,
    normalization: str = "imagenet",
):
    transforms = [A.Resize(image_size, image_size)]
    if use_horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    geometry_transform = (
        A.Affine(scale=(0.92, 1.08), translate_percent=(-0.05, 0.05), rotate=(-12, 12), p=0.5)
        if use_affine
        else A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=12, p=0.5)
    )
    transforms.extend(
        [
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
            geometry_transform,
            _build_normalize(normalization),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_eval_transform(image_size: int, normalization: str = "imagenet"):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            _build_normalize(normalization),
            ToTensorV2(),
        ]
    )
