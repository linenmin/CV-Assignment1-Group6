"""
对 neighborhood_aware 推理的 top_k 与 base_weight 做离线网格扫描。

一次加载 checkpoint 与 embedding，在内存中遍历参数网格。

**必须与 baseline submission 按 id 对齐后**再统计类别迁移（不可假设 DataLoader 行序等于 submission 排序）。

推荐配置以 **submission 差异画像** 为主（先抑制危险的 other→目标类 放宽），验证集准确率仅作旁注，不作主排序依据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_script_dir))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import project_path
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.inference.prototype import (
    combine_embedding_sets,
    compute_class_prototypes,
    neighborhood_aware_predictions,
)
from dl_pipeline.training.lightning_module import FaceClassifierModule

# 与 predict.py 保持一致
from predict import _collect_embeddings  # noqa: E402


def _count_transitions(baseline: pd.Series, candidate: pd.Series) -> dict[str, int]:
    """统计 baseline -> candidate 的类别迁移计数。"""
    keys = ["0_to_1", "0_to_2", "1_to_0", "2_to_0", "1_to_2", "2_to_1", "unchanged"]
    counts = {k: 0 for k in keys}
    for a, b in zip(baseline.tolist(), candidate.tolist()):
        if a == b:
            counts["unchanged"] += 1
        else:
            counts[f"{a}_to_{b}"] += 1
    return counts


def _pick_winner_submission_first(rows: list[dict], ref_top_k: int = 15, ref_base_weight: float = 0.5) -> dict:
    """
    以相对 baseline submission 的迁移为主（与 task 上「小验证集不可靠」一致）：

    1) 最小化 (0->1 + 0->2)：历史上大规模 other→目标 与掉分强相关；
    2) 并列时最小化 total_changed；
    3) 再并列：与 exp_047 超参 (ref_top_k, ref_base_weight) 的距离最小，便于解释。
    """
    best = None
    for row in rows:
        if best is None:
            best = row
            continue
        relax = row["0_to_1"] + row["0_to_2"]
        best_relax = best["0_to_1"] + best["0_to_2"]
        if relax < best_relax:
            best = row
            continue
        if relax > best_relax:
            continue
        if row["total_changed"] < best["total_changed"]:
            best = row
            continue
        if row["total_changed"] > best["total_changed"]:
            continue
        dist = abs(row["top_k"] - ref_top_k) + abs(row["base_weight"] - ref_base_weight) * 10.0
        best_dist = abs(best["top_k"] - ref_top_k) + abs(best["base_weight"] - ref_base_weight) * 10.0
        if dist < best_dist:
            best = row
    return best


def _transition_counts_vs_baseline(
    baseline_df: pd.DataFrame,
    test_ids: torch.Tensor,
    test_pred: torch.Tensor,
) -> dict[str, int]:
    """按 id 与 baseline submission 对齐后统计迁移。"""
    cand = pd.DataFrame(
        {
            "id": test_ids.detach().cpu().numpy().ravel().astype(int),
            "class_c": test_pred.detach().cpu().numpy().ravel().astype(int),
        }
    )
    merged = baseline_df.merge(cand, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(baseline_df):
        raise RuntimeError(
            f"baseline 与候选预测 id 无法一一对齐: baseline={len(baseline_df)}, merged={len(merged)}"
        )
    return _count_transitions(merged["class"], merged["class_c"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_047_vit_adaface_haar_1ep_frozen_neighborhoodaware_fixed055_hfliptta.yaml",
        help="与 exp_047 相同的实验配置（数据、模型、TTA）",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="若不指定则使用 exp_032 的 best.ckpt（与 exp_047 一致）",
    )
    parser.add_argument(
        "--baseline-submission",
        default="data/submissions/20260402_164731_exp_047_vit_adaface_haar_1ep_frozen_neighborhoodaware_fixed055_hfliptta_submission.csv",
    )
    parser.add_argument(
        "--output-json",
        default="reports/sweeps/neighborhood_aware_grid_latest.json",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    splits_dir = project_path(config["data"]["splits_dir"])

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(
            project_path(
                "outputs",
                "exp_032_vit_adaface_haar_1ep_frozen_prototype_fixed055",
                "checkpoints",
                "best.ckpt",
            )
        )

    inference_config = config.get("inference", {})
    prototype_labels = inference_config.get("prototype_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold = inference_config.get("threshold", 0.55)
    use_tta = inference_config.get("tta_horizontal_flip", False)

    train_df = pd.read_csv(splits_dir / "train.csv")
    loss_config = config.get("loss", {})
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["train"].get("accelerator") != "cpu" else "cpu"
    )

    datamodule = FaceDataModule(
        train_csv=splits_dir / "train.csv",
        val_csv=splits_dir / "val.csv",
        test_csv=splits_dir / "test.csv",
        image_size=config["data"]["face_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        normalization=config["data"]["normalization"],
    )
    datamodule.setup()

    model = FaceClassifierModule.load_from_checkpoint(
        checkpoint_path,
        model_family=config["model"]["family"],
        backbone_name=config["model"]["backbone_name"],
        num_classes=int(train_df["class"].nunique()),
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        learning_rate=config["train"]["learning_rate"],
        backbone_learning_rate=config["train"].get("backbone_learning_rate"),
        weight_decay=config["train"]["weight_decay"],
        scheduler_name=config["train"]["scheduler"],
        max_epochs=config["train"]["max_epochs"],
        pretrained_repo_id=config["model"].get("pretrained_repo_id"),
        pretrained_checkpoint_path=config["model"].get("pretrained_checkpoint_path"),
        freeze_backbone=config["model"].get("freeze_backbone", False),
        unfreeze_last_stage=config["model"].get("unfreeze_last_stage", False),
        unfreeze_stage_count=config["model"].get("unfreeze_stage_count", 0),
        unfreeze_cvlface_norm=config["model"].get("unfreeze_cvlface_norm", True),
        unfreeze_cvlface_feature=config["model"].get("unfreeze_cvlface_feature", True),
        loss_name=loss_config.get("name", "cross_entropy"),
        loss_target_labels=loss_config.get("target_labels"),
        arcface_scale=loss_config.get("arcface_scale", 30.0),
        arcface_margin=loss_config.get("arcface_margin", 0.5),
        cosface_scale=loss_config.get("cosface_scale", loss_config.get("arcface_scale", 30.0)),
        cosface_margin=loss_config.get("cosface_margin", 0.35),
    )
    model = model.to(device)
    model.eval()

    train_emb, train_lab = _collect_embeddings(
        model, datamodule.train_dataloader(), device, use_horizontal_flip_tta=use_tta
    )
    val_emb, val_lab = _collect_embeddings(
        model, datamodule.val_dataloader(), device, use_horizontal_flip_tta=use_tta
    )
    test_emb, test_ids = _collect_embeddings(
        model, datamodule.predict_dataloader(), device, use_horizontal_flip_tta=use_tta
    )

    proto_train = compute_class_prototypes(train_emb, train_lab, prototype_labels=prototype_labels)
    all_emb, all_lab = combine_embedding_sets([(train_emb, train_lab), (val_emb, val_lab)])
    proto_all = compute_class_prototypes(all_emb, all_lab, prototype_labels=prototype_labels)

    baseline_path = project_path(args.baseline_submission)
    baseline_df = pd.read_csv(baseline_path).sort_values("id").reset_index(drop=True)

    top_k_list = [5, 10, 15, 20, 30]
    base_weight_list = [0.3, 0.4, 0.5, 0.6, 0.7]

    rows: list[dict] = []
    for top_k in top_k_list:
        for base_weight in base_weight_list:
            val_pred, _, _, _ = neighborhood_aware_predictions(
                query_embeddings=val_emb,
                prototypes=proto_train,
                other_label=other_label,
                threshold=threshold,
                top_k=top_k,
                base_weight=base_weight,
            )
            val_accuracy = (val_pred == val_lab).float().mean().item()

            test_pred, _, _, _ = neighborhood_aware_predictions(
                query_embeddings=test_emb,
                prototypes=proto_all,
                other_label=other_label,
                threshold=threshold,
                top_k=top_k,
                base_weight=base_weight,
            )
            trans = _transition_counts_vs_baseline(baseline_df, test_ids, test_pred)
            total_changed = len(baseline_df) - trans["unchanged"]
            rows.append(
                {
                    "top_k": top_k,
                    "base_weight": base_weight,
                    "neighbor_weight": round(1.0 - base_weight, 4),
                    "val_accuracy": round(val_accuracy, 6),
                    "total_changed": total_changed,
                    "0_to_1": trans["0_to_1"],
                    "0_to_2": trans["0_to_2"],
                    "1_to_0": trans["1_to_0"],
                    "2_to_0": trans["2_to_0"],
                    "1_to_2": trans["1_to_2"],
                    "2_to_1": trans["2_to_1"],
                }
            )

    winner = _pick_winner_submission_first(rows, ref_top_k=15, ref_base_weight=0.5)
    out_path = project_path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": checkpoint_path,
        "baseline_submission": str(baseline_path),
        "threshold": threshold,
        "winner_selection": "submission_diff_first_minimize_other_to_target_then_total_changed_then_near_exp047",
        "winner": winner,
        "grid": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(winner, indent=2, ensure_ascii=False))
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
