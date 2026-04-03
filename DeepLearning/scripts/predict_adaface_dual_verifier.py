"""
exp_068: 在现有 AdaFace ViT embedding 上跑双独立验证器 + look-alike 阈值校准。
零新模型、零微调——只改推理逻辑。

用法（在 DeepLearning 目录下）：
  python scripts/predict_adaface_dual_verifier.py \
    --checkpoint outputs/exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta/checkpoints/best.ckpt \
    --config configs/experiments/exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml \
    --exp_name exp_068_adaface_vit_dual_verifier_lookalike \
    --top_k 5 \
    --tta_hflip
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import project_path
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.training.lightning_module import FaceClassifierModule

# 复用 dual verifier 核心逻辑
from dl_pipeline.inference.insightface_dual_verifier import (
    dual_verifier_predict,
    loo_top_k_mean_for_identity,
    top_k_mean_cosine_similarity,
    youden_threshold,
)


def _load_model(config, ckpt_path: str):
    train_df = pd.read_csv(project_path(config["data"]["splits_dir"]) / "train.csv")
    loss_config = config.get("loss", {})
    model = FaceClassifierModule.load_from_checkpoint(
        ckpt_path,
        model_family=config["model"]["family"],
        backbone_name=config["model"]["backbone_name"],
        num_classes=int(train_df["class"].nunique()),
        pretrained=config["model"]["pretrained"],
        dropout=config["model"].get("dropout", 0.0),
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
    model.eval()
    return model


@torch.no_grad()
def _extract_all_embeddings_simple(
    model, device, config, train_df, val_df, test_df, tta_hflip: bool
) -> dict[int, np.ndarray]:
    """逐图提取 embedding，不依赖 DataModule。"""
    import cv2
    from dl_pipeline.data.transforms import build_eval_transform

    transform = build_eval_transform(
        config["data"].get("face_size", 112),
        config["data"].get("normalization", "face"),
    )
    id_to_emb: dict[int, np.ndarray] = {}

    all_rows = pd.concat([train_df, val_df, test_df], ignore_index=True)
    for _, row in all_rows.iterrows():
        sid = int(row["id"])
        if sid in id_to_emb:
            continue
        img_path = project_path(row["image_path"])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

        feats = model.extract_features(tensor)
        if tta_hflip:
            tensor_f = torch.flip(tensor, dims=[3])
            feats_f = model.extract_features(tensor_f)
            feats = feats + feats_f
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        id_to_emb[sid] = feats[0].cpu().numpy()

    return id_to_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp_name", default="exp_068_adaface_vit_dual_verifier_lookalike")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--tta_hflip", action="store_true")
    parser.add_argument("--lookalike_csv", default="reports/analysis/exp_029_other_k2/other_cluster_assignments.csv")
    parser.add_argument("--michael_cluster", type=int, default=1)
    parser.add_argument("--sarah_cluster", type=int, default=0)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"加载模型: {args.checkpoint}")
    model = _load_model(config, args.checkpoint).to(device)

    splits_dir = project_path(config["data"]["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    print(f"提取全量 embedding (tta_hflip={args.tta_hflip}) ...")
    id_to_emb = _extract_all_embeddings_simple(
        model, device, config, train_df, val_df, test_df, args.tta_hflip
    )
    print(f"  共 {len(id_to_emb)} 个 embedding")

    # 合并 train+val 作为完整标注集
    all_labeled = pd.concat([train_df, val_df], ignore_index=True)

    # 构建 gallery
    def stack_class(df, cls):
        return [id_to_emb[int(r["id"])] for _, r in df.iterrows()
                if int(r["class"]) == cls and int(r["id"]) in id_to_emb]

    jesse_all = stack_class(all_labeled, 1)
    mila_all = stack_class(all_labeled, 2)
    jesse_gallery = np.stack(jesse_all) if jesse_all else np.zeros((0, 512), dtype=np.float32)
    mila_gallery = np.stack(mila_all) if mila_all else np.zeros((0, 512), dtype=np.float32)

    print(f"  Jesse gallery: {len(jesse_all)}, Mila gallery: {len(mila_all)}")

    # look-alike 校准
    look_df = pd.read_csv(project_path(args.lookalike_csv))
    all_other_ids = all_labeled[all_labeled["class"] == 0]["id"].astype(int).tolist()
    michael_ids = [i for i in look_df[look_df["cluster"] == args.michael_cluster]["id"].astype(int).tolist()
                   if i in all_other_ids]
    sarah_ids = [i for i in look_df[look_df["cluster"] == args.sarah_cluster]["id"].astype(int).tolist()
                 if i in all_other_ids]
    michael_embs = [id_to_emb[i] for i in michael_ids if i in id_to_emb]
    sarah_embs = [id_to_emb[i] for i in sarah_ids if i in id_to_emb]

    print(f"  Michael 校准: {len(michael_embs)}, Sarah 校准: {len(sarah_embs)}")

    # Jesse 通道校准
    scores_j_pos = np.array([loo_top_k_mean_for_identity(i, jesse_all, args.top_k)
                             for i in range(len(jesse_all))])
    scores_j_neg = np.array([top_k_mean_cosine_similarity(e, jesse_gallery, args.top_k)
                             for e in michael_embs])
    theta_jesse, meta_j = youden_threshold(scores_j_pos, scores_j_neg)

    # Mila 通道校准
    scores_m_pos = np.array([loo_top_k_mean_for_identity(i, mila_all, args.top_k)
                             for i in range(len(mila_all))])
    scores_m_neg = np.array([top_k_mean_cosine_similarity(e, mila_gallery, args.top_k)
                             for e in sarah_embs])
    theta_mila, meta_m = youden_threshold(scores_m_pos, scores_m_neg)

    print(f"  theta_jesse = {theta_jesse:.4f}, theta_mila = {theta_mila:.4f}")

    # 验证集准确率
    correct = total = 0
    for _, row in val_df.iterrows():
        sid = int(row["id"])
        if sid not in id_to_emb:
            continue
        probe = id_to_emb[sid]
        sj = top_k_mean_cosine_similarity(probe, jesse_gallery, args.top_k)
        sm = top_k_mean_cosine_similarity(probe, mila_gallery, args.top_k)
        pred = dual_verifier_predict(sj, sm, theta_jesse, theta_mila)
        if pred == int(row["class"]):
            correct += 1
        total += 1
    val_acc = correct / total if total else None
    print(f"  val_accuracy = {val_acc}")

    # 测试集预测
    predictions = []
    for _, row in test_df.iterrows():
        sid = int(row["id"])
        if sid not in id_to_emb:
            predictions.append(0)
            continue
        probe = id_to_emb[sid]
        sj = top_k_mean_cosine_similarity(probe, jesse_gallery, args.top_k)
        sm = top_k_mean_cosine_similarity(probe, mila_gallery, args.top_k)
        predictions.append(dual_verifier_predict(sj, sm, theta_jesse, theta_mila))

    dist = Counter(predictions)
    print(f"  预测分布: {dict(sorted(dist.items()))}")

    # 保存 submission
    submission = pd.DataFrame({"id": test_df["id"], "class": predictions})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = project_path("data", "submissions")
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_path = sub_dir / f"{ts}_{args.exp_name}_submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  submission: {sub_path}")

    # 保存指标
    out_dir = project_path("outputs", args.exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mode": "adaface_dual_verifier",
        "checkpoint": args.checkpoint,
        "gallery_top_k": args.top_k,
        "tta_horizontal_flip": args.tta_hflip,
        "theta_jesse": theta_jesse,
        "theta_mila": theta_mila,
        "calib_meta_jesse": meta_j,
        "calib_meta_mila": meta_m,
        "num_jesse_gallery": len(jesse_all),
        "num_mila_gallery": len(mila_all),
        "num_michael_calib": len(michael_embs),
        "num_sarah_calib": len(sarah_embs),
        "val_accuracy": val_acc,
        "test_distribution": dict(sorted(dist.items())),
    }
    (out_dir / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 与 exp_061 对比
    import glob
    subs61 = sorted(glob.glob(str(project_path("data", "submissions", "*exp_061*"))))
    if subs61:
        s061 = pd.read_csv(subs61[-1])
        m = submission.merge(s061, on="id", suffixes=("_new", "_061"))
        d = m[m["class_new"] != m["class_061"]]
        print(f"\n  vs exp_061: {len(d)} diffs")
        for (f, t), c in sorted(Counter(zip(d["class_061"], d["class_new"])).items()):
            print(f"    {f}->{t}: {c}")


if __name__ == "__main__":
    main()
