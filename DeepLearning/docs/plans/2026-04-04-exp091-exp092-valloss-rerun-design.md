# exp_091 / exp_092 Val-Loss Rerun Design

**Goal:** 修正当前冠军主线及其 stronger-finetune 分支的训练协议，把早停与最佳 checkpoint 选择从 `val_acc` 改为 `val_loss`，并把 `max_epochs` 提升到 `50`，重新验证先前结论是否因小验证集过早饱和而失真。

**Architecture:** 不改变输入协议、模型族、loss、推理协议，只修正训练监控信号。`exp_091` 复现 `exp_088` 主线，`exp_092` 复现 `exp_090` stronger-finetune 分支；两者都改为 `monitor_metric: val_loss`、`monitor_mode: min`、更长训练上限与更宽松 patience。

**Tech Stack:** `cvlface AdaFace ViT`, Lightning, 现有 `train.py` / `predict.py`

## Chosen Scope

- 不改：
  - `multiface-selected` 输入协议
  - backbone / finetune 深度（分别继承 `exp_088` 与 `exp_090`）
  - CE loss
  - neighborhood-aware 推理协议
- 只改：
  - `monitor_metric: val_acc -> val_loss`
  - `monitor_mode: max -> min`
  - `max_epochs: 20 -> 50`
  - `early_stopping_patience: 6 -> 12`

## Why This Rerun Is Mandatory

- 当前 val 只有 `16` 张
- `val_acc` 的分辨率只有 `1/16`
- 一旦很早达到 `1.0`，`val_acc` 就不再提供有效训练进度信号
- 但日志已经显示 `val_loss` 在 `val_acc=1.0` 之后仍持续下降

因此：

- 旧的 `exp_088` / `exp_090` 结论仍有参考价值
- 但尚未被完全公平地验证

## Expected Outcome

- 如果 `exp_091` 或 `exp_092` 的 submission 与旧版不同，说明原先结论确实受到了不合理早停影响
- 如果仍然逐行一致，才能更有把握地说：
  - `exp_088` 已经稳定在当前局部最优附近
  - `exp_090` 的 stronger finetune 方向当前确实未带来额外收益

## Actual Outcome

- `exp_091` 与 `exp_088`：`num_diff = 0`
- `exp_092` 与 `exp_088`：`num_diff = 0`
- `exp_091` 最佳 `val_loss = 0.0025814103428274393`
- `exp_092` 最佳 `val_loss = 0.003377359127625823`

## What Changed And What Did Not

- 变了：
  - 训练 protocol 被纠正了
  - 两个实验都不再在早期因 `val_acc=1.0` 停住
  - 两者都继续训练到了接近 `50 epoch`
  - 这直接证明用户对旧 protocol 的批评成立
- 没变：
  - `exp_091` 没有改写 `exp_088`
  - `exp_092` 没有改写 `exp_090` 相对 `exp_088` 的结论

## Final Decision

- 后续冠军主线默认使用 `val_loss / min` 做 monitor
- `exp_088` 仍是当前稳定主线
- `exp_090` 的 stronger finetune 方向在当前输入协议下暂不继续细抠
