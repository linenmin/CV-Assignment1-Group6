# exp_088 Multiface Train Design

**Goal:** 基于 `exp_061` 的训练配方，仅替换训练/验证/测试输入为“多脸自动选中的单脸”，验证输入质量修复是否能提升可训练分类主线。

**Architecture:** 先从原图中检测所有脸，使用当前 `exp_086/087` 的双通道打分逻辑为每张脸计算 `class1/class2` 相似度，再按样本标签或无标签规则选出一张脸，生成新的 processed dataset 与 split CSV。训练与预测仍复用现有 `train.py` / `predict.py`。

**Tech Stack:** `insightface buffalo_l`, OpenCV, pandas, Lightning, 现有 `FaceDataModule` / `FaceClassifierModule`

## Chosen Scope

- 基线：`exp_061`
- 不改：
  - backbone
  - loss
  - epoch
  - 解冻策略
  - neighborhood-aware 推理协议
- 只改：
  - processed dataset 的脸选择逻辑
  - 明确坏图剔除（首版仅 `id=65`）

## Face Selection Rules

- `class1` 样本：选 `class1` 分数最高的脸
- `class2` 样本：选 `class2` 分数最高的脸
- `class0` 样本：选 `max(class1_excess, class2_excess)` 最高的脸，作为 hardest negative
- test 样本：同样选 `max(class1_excess, class2_excess)` 最高的脸
- 若原图检测不到脸：回退到原有 `image_path` 的 HAAR crop

## First Experiment

- 新 processed_dir：`data/processed/exp_088_multiface_selected_faces_224`
- 新 splits_dir：`data/splits/exp_088_multiface_selected`
- 新 config：`exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta`
- 训练后直接生成 submission，与 `exp_061 / exp_086 / exp_087` 对比

## Expected Outcome

- 如果 `exp_088` 提升，说明此前可训练分类线的主要瓶颈确实是输入脸选择错误
- 如果 `exp_088` 不提升而 `exp_086/087` 仍强，则说明“多脸选人”更适合 verifier 路线而不一定直接迁移到分类训练

## Actual Outcome

- `exp_088` public leaderboard score：`0.98403`
- 结果高于：
  - `exp_087 = 0.97191`
  - `exp_086 = 0.97026`
  - `exp_082 = 0.94768`
- 该结果已经验证：
  - “多脸自动选人”不只是推理层补丁
  - 它可以直接作为训练输入协议，显著提升可训练分类主线

## Updated Plan

- 将 `multiface-selected` 输入协议视为新的主线数据入口
- 后续实验优先围绕这条输入协议做增量优化，而不是回到单脸 `HAAR` 训练线
- 优先检查还能否在以下方向继续抬高上限：
  - 更强 backbone / 更长训练在同一输入协议下是否继续收益
  - `class0` hardest negative 采样是否还能更稳
  - `crop_fallback` 样本是否还能进一步减少
