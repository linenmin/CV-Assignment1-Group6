# exp_089 Class0 Hard Negatives Design

**Goal:** 在保持 `exp_088` 输入协议与训练超参不变的前提下，仅扩展训练集 `class0` 的 hardest negative 人脸采样，验证负样本建模是否还能继续抬高线上分数。

**Architecture:** 继续使用 `multiface-selected` 输入协议。`class1/2` 与 val/test 保持单脸选择；仅在 train split 的 `class0` 样本上，按目标类相似度排序保留前 `k` 张最危险的人脸，并展开为多条训练样本。训练与预测仍复用现有 `train.py` / `predict.py`。

**Tech Stack:** `insightface buffalo_l`, OpenCV, pandas, Lightning, 现有 `FaceDataModule` / `FaceClassifierModule`

## Chosen Scope

- 基线：`exp_088`
- 不改：
  - backbone
  - loss
  - epoch
  - 解冻策略
  - neighborhood-aware 推理协议
  - `class1/2` 的多脸选人规则
  - val/test 的单脸输入协议
- 只改：
  - train split 中 `class0` 的样本展开逻辑
  - 新 config / processed dataset / split 输出

## Face Selection Rules

- `class1` train/val：仍取 `class1` 分数最高的人脸
- `class2` train/val：仍取 `class2` 分数最高的人脸
- `class0` train：取 `max(excess_jesse, excess_mila)` 最高的前 `k` 张脸，按 rank 展开为多行训练样本
- test：仍只取最强单脸，不展开
- 若原图检测不到脸：仍回退到原有 `image_path` 的 HAAR crop

## Experiment

- 新 processed_dir：`data/processed/exp_089_multiface_selected_other_top2_faces_112`
- 新 splits_dir：`data/splits/exp_089_multiface_selected_other_top2`
- 新 config：`exp_089_vit_adaface_multiface_selected_other_top2_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta`
- 首轮固定 `class0 top-k = 2`

## Expected Outcome

- 如果 `exp_089` 提升，说明在“选对脸”之后，下一瓶颈主要是 `other` 的 hard negative 建模
- 如果 `exp_089` 不提升甚至下降，说明 `exp_088` 当前的单 hardest negative 已足够，后续应转向更强 backbone / 更长训练，而不是继续扩展 `class0`

## Actual Outcome

- 新 train split 从 `63` 扩到 `66`
- 实际只有 `3` 个 `class0` 样本被展开为 `top-2` negatives
- `best_val_acc = 1.0`
- `exp_089` submission 与 `exp_088` **逐行完全一致**
  - `num_diff = 0`

## Updated Plan

- 不继续沿 `class0 top-k negatives` 做更细碎的展开微调
- 下一优先级转向：
  - 同一输入协议下提升训练强度
  - 或继续压缩 `crop_fallback`
