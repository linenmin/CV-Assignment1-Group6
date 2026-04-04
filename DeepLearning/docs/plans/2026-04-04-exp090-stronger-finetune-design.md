# exp_090 Stronger Finetune Design

**Goal:** 在保持 `exp_088` 的 `multiface-selected` 输入协议、分类目标和推理协议不变的前提下，仅增强 ViT 主干的可学习深度，验证单体模型在干净输入下是否还能继续提升。

**Architecture:** 继续复用 `exp_088` 的 processed dataset 与 split。模型仍为 `AdaFace ViT + CE + neighborhood-aware`，仅把 backbone 解冻深度从最后 `2` 个 block 提升到最后 `4` 个 block，并将 backbone 学习率下调，控制更深 finetune 的稳定性。

**Tech Stack:** `cvlface AdaFace ViT`, Lightning, 现有 `train.py` / `predict.py`

## Chosen Scope

- 基线：`exp_088`
- 不改：
  - `multiface-selected` 输入协议
  - CE loss
  - train/val/test split
  - neighborhood-aware 推理协议
  - threshold / TTA
- 只改：
  - `unfreeze_stage_count: 2 -> 4`
  - `backbone_learning_rate: 1e-5 -> 5e-6`

## Why This First

- `exp_089` 已经证明：继续在 `class0 top-k negatives` 这条线上小修补，无法改变最终 submission
- `exp_088` 真正打开的是“干净输入下单体模型还有多少训练增益”这个问题
- 因此最合理的下一步不是再动数据协议，而是直接检查更深层 finetune 是否能带来新的可学习容量

## Expected Outcome

- 如果 `exp_090` 提升，说明当前主瓶颈已经转移到单体模型训练强度
- 如果 `exp_090` 不提升，则说明 `exp_088` 已经接近这条 CE ViT 主线的局部天花板，后续应考虑更换训练目标或进一步处理 fallback

## Actual Outcome

- `best_val_acc = 1.0`
- `exp_090` submission 与 `exp_088` **逐行完全一致**
  - `num_diff = 0`

## Updated Plan

- 不继续沿 `last2 -> last4 -> last6` 这种同轴加深 finetune 方向细抠
- 下一步优先考虑：
  - 改训练目标
  - 或处理剩余 `crop_fallback`
