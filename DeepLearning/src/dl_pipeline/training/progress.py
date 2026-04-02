from __future__ import annotations

import sys

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm


class AsciiTQDMProgressBar(TQDMProgressBar):
    """Windows / IDE 终端友好的进度条。

    说明：

    1. Lightning 2.6 在已安装 ``ipywidgets`` 时会走 ``tqdm.auto``，在 Cursor/部分 IDE
       集成终端（非 TTY 或伪终端）里，进度条可能不刷新或看起来像「消失」。
    2. 旧实现先 ``super().init_train_tqdm()`` 再 ``setattr(bar, "ascii", True)``：
       许多 tqdm 版本在创建后再改 ``ascii`` 不会正确重绘，导致与「以前能看到的条」不一致。

    因此在构造 Lightning 的 ``Tqdm`` 时直接传入 ``ascii=True``（与父类参数对齐），
    保证与仓库 ``findings.md`` / README 中「关闭 rich、用 tqdm」的设计一致。
    """

    def __init__(self, refresh_rate: int = 1) -> None:
        super().__init__(refresh_rate=refresh_rate)

    def init_sanity_tqdm(self):
        return Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
            ascii=True,
        )

    def init_train_tqdm(self):
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
            ascii=True,
        )

    def init_validation_tqdm(self):
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
            ascii=True,
        )

    def init_predict_tqdm(self):
        return Tqdm(
            desc=self.predict_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
            ascii=True,
        )

    def init_test_tqdm(self):
        return Tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
            ascii=True,
        )
