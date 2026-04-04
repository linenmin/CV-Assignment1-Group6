# exp_097 CosFace Retrain Design

**Goal:** 在当前最干净的 `exp_093` rescue 输入协议上，做一版与 `exp_096` 完全受控对比的 CosFace 重训，验证 margin-based loss 是否能比 CE 更好地塑造 embedding 空间，并推动当前残余弱证据错例。

**Architecture:** 训练/推理协议整体继承 `exp_096`，只把训练目标从 `cross_entropy` 切换到三类 `cosface`。保持 backbone、输入协议、finetune 深度、epoch、monitor、推理协议不变。

## Why This Is The Right Next Step

- 当前系统最终决策依赖：
  - backbone embedding
  - prototype / neighborhood-aware
  - open-set rejection
- 这本质上是一个 embedding 几何问题，而不仅是普通 softmax 分类问题
- `CosFace` 比 `CE` 更直接地优化：
  - 类内紧致
  - 类间角度间隔
- 早期 CosFace 实验发生在错误数据处理时代，因此不能作为反证

## Chosen Scope

- 固定：
  - `exp_093` input protocol
  - `AdaFace ViT`
  - `last2block`
  - `50 epoch`
  - `val_loss / min`
  - `neighborhood_aware`
  - `hfliptta`
- 唯一改变：
  - `loss.name: cross_entropy -> cosface`
  - `cosface_scale: 30.0`
  - `cosface_margin: 0.35`

## Why Not Add Sample Weighting / Handcrafted Rules

- 这轮目标是保持方法论优雅和报告可写性
- 不引入：
  - 手工 mode 标签
  - class-specific sample weighting
  - leaderboard-targeted post-hoc patch

## Decision Rule

- 如果 `exp_097` 相比 `exp_096 / exp_088` 产生有意义的 test 侧变化，尤其修复用户已确认的弱证据样本，则优先提交
- 如果仍然逐行一致，则说明“margin-based embedding 几何”也不是当前单体模型的突破点，后续再考虑更结构化的 metric-learning 或专门的 fallback-target training strategy
