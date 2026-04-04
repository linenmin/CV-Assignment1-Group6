# exp_096 Pad18-Rescue Retrain Design

**Goal:** 在 `exp_093` 的 `pad18 rescue multiface-selected` 输入协议上，重新训练当前冠军单体模型主线，验证训练分布与推理分布重新对齐后，是否能系统性抬升弱证据 target 样本的分数。

**Architecture:** 训练 recipe 直接继承 `exp_091`，只替换输入协议与对应的数据目录。保持 `AdaFace ViT + last2block + CE + neighborhood-aware + hfliptta`，monitor 使用 `val_loss / min`，训练上限 `50 epoch`。

**Why This Is The Right Next Step:**

- `exp_094/095` 已证明阈值不是当前主瓶颈
- 当前高价值错例不是 identity 方向判错，而是 target 证据过弱
- 当前最明显的不一致是：
  - 模型训练于 `exp_088` 输入协议
  - 但后续排错与推理修补已经开始依赖 `exp_093` rescue 输入协议
- 因此最自然、最干净的根治路径，是在 `exp_093` 输入协议上重训，而不是继续 patch submission

## Chosen Scope

- 继承：
  - `exp_091` 的训练监控协议
  - `50 epoch`
  - `val_loss / min`
  - `last2block`
  - `CE`
  - `neighborhood_aware`
  - `tta_horizontal_flip`
- 替换：
  - `processed_dir -> exp_093_multiface_selected_pad18_rescue_faces_112`
  - `splits_dir -> exp_093_multiface_selected_pad18_rescue`
  - `multiface_selection.rescue_pad_fractions -> [0.18]`

## Success Criteria

- 训练正常收敛，产出稳定 checkpoint
- 与 `exp_088/091` 相比，submission 出现有意义变化
- 尤其关注此前用户肉眼确认的高价值弱证据样本：
  - `48, 289, 361, 543, 612, 699, 1333, 1525`

## Decision Rule

- 若 `exp_096` 改善 submission 且方向符合肉眼审计，则优先提交
- 若 `exp_096` 仍与 `exp_088/091` 高度一致，则说明仅靠输入协议对齐仍不足以根治，之后再考虑更强训练目标或专门针对 fallback 样本的训练增强
