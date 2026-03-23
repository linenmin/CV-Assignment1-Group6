from __future__ import annotations

import sys

from lightning.pytorch.callbacks import TQDMProgressBar


class AsciiTQDMProgressBar(TQDMProgressBar):
    """Windows 终端友好的 ASCII 进度条。"""

    def __init__(self, refresh_rate: int = 1) -> None:
        super().__init__(refresh_rate=refresh_rate)

    def _common_kwargs(self) -> dict:
        return {
            "ascii": True,
            "file": sys.stdout,
            "leave": True,
        }

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        for key, value in self._common_kwargs().items():
            setattr(bar, key, value)
        return bar

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        for key, value in self._common_kwargs().items():
            setattr(bar, key, value)
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        for key, value in self._common_kwargs().items():
            setattr(bar, key, value)
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        for key, value in self._common_kwargs().items():
            setattr(bar, key, value)
        return bar
