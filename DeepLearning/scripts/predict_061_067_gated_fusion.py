from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.inference.gated_fusion import (
    apply_gated_anchor_fusion,
    select_best_gated_fusion_params,
    summarize_dual_verifier_scores,
)
from dl_pipeline.inference.insightface_dual_verifier import (
    _load_lookalike_clusters,
    _read_bgr,
    loo_top_k_mean_for_identity,
    top_k_mean_cosine_similarity,
    youden_threshold,
)
from dl_pipeline.inference.prototype import (
    combine_embedding_sets,
    compute_class_prototypes,
    neighborhood_aware_predictions,
)
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe
from dl_pipeline.inference.tta import extract_tta_features
from dl_pipeline.training.lightning_module import FaceClassifierModule


def _resolve_checkpoint_path(config: dict[str, Any], explicit_path: str | None) -> Path:
    if explicit_path:
        return project_path(explicit_path)
    ckpt_dir = project_path("outputs", config["experiment_name"], "checkpoints")
    best_ckpt = ckpt_dir / "best.ckpt"
    if best_ckpt.is_file():
        return best_ckpt
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt
    raise FileNotFoundError(f"找不到 checkpoint: {ckpt_dir}")


def _load_model_from_checkpoint(config: dict[str, Any], checkpoint_path: Path) -> FaceClassifierModule:
    train_df = pd.read_csv(project_path(config["data"]["splits_dir"]) / "train.csv")
    loss_config = config.get("loss", {})
    return FaceClassifierModule.load_from_checkpoint(
        str(checkpoint_path),
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


def _collect_embeddings(model, dataloader, device, use_horizontal_flip_tta: bool = False):
    embeddings = []
    values = []
    model.eval()
    with torch.no_grad():
        for images, batch_values in dataloader:
            features = extract_tta_features(
                model=model,
                images=images,
                device=device,
                use_horizontal_flip=use_horizontal_flip_tta,
            )
            embeddings.append(features)
            values.append(batch_values.detach().cpu())
    return torch.cat(embeddings, dim=0), torch.cat(values, dim=0)


def _build_datamodule(config: dict[str, Any]) -> FaceDataModule:
    return FaceDataModule(
        train_csv=project_path(config["data"]["splits_dir"]) / "train.csv",
        val_csv=project_path(config["data"]["splits_dir"]) / "val.csv",
        test_csv=project_path(config["data"]["splits_dir"]) / "test.csv",
        image_size=config["data"]["face_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_horizontal_flip=config["augmentation"]["use_horizontal_flip"],
        use_affine=config["augmentation"]["use_affine"],
        use_degradation_pack=config["augmentation"].get("use_degradation_pack", False),
        normalization=config["data"]["normalization"],
    )


def _score_neighborhood_tables(
    config: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    datamodule = _build_datamodule(config)
    datamodule.setup()

    splits_dir = project_path(config["data"]["splits_dir"])
    val_meta = pd.read_csv(splits_dir / "val.csv")
    test_meta = pd.read_csv(splits_dir / "test.csv")

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"].get("accelerator", "gpu") != "cpu" else "cpu")
    model = _load_model_from_checkpoint(config, checkpoint_path).to(device)

    use_horizontal_flip_tta = bool(config.get("inference", {}).get("tta_horizontal_flip", False))
    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device, use_horizontal_flip_tta)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device, use_horizontal_flip_tta)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device, use_horizontal_flip_tta)

    inf = config["inference"]
    prototype_labels = inf.get("prototype_labels", [1, 2])
    other_label = int(inf.get("other_label", 0))
    threshold = float(inf.get("threshold", 0.55))
    nb = inf.get("neighborhood_aware", {})
    top_k = int(nb.get("top_k", 15))
    base_weight = float(nb.get("base_weight", 0.5))

    train_prototypes = compute_class_prototypes(train_embeddings, train_labels, prototype_labels=prototype_labels)
    val_argmax, val_base, val_neighbor, val_final = neighborhood_aware_predictions(
        query_embeddings=val_embeddings,
        prototypes=train_prototypes,
        other_label=other_label,
        threshold=-1e9,
        top_k=top_k,
        base_weight=base_weight,
    )
    val_open, _, _, _ = neighborhood_aware_predictions(
        query_embeddings=val_embeddings,
        prototypes=train_prototypes,
        other_label=other_label,
        threshold=threshold,
        top_k=top_k,
        base_weight=base_weight,
    )

    all_gallery_embeddings, all_gallery_labels = combine_embedding_sets(
        [(train_embeddings, train_labels), (val_embeddings, val_labels)]
    )
    all_prototypes = compute_class_prototypes(all_gallery_embeddings, all_gallery_labels, prototype_labels=prototype_labels)
    test_argmax, test_base, test_neighbor, test_final = neighborhood_aware_predictions(
        query_embeddings=test_embeddings,
        prototypes=all_prototypes,
        other_label=other_label,
        threshold=-1e9,
        top_k=top_k,
        base_weight=base_weight,
    )
    test_open, _, _, _ = neighborhood_aware_predictions(
        query_embeddings=test_embeddings,
        prototypes=all_prototypes,
        other_label=other_label,
        threshold=threshold,
        top_k=top_k,
        base_weight=base_weight,
    )

    val_table = pd.DataFrame(
        {
            "id": val_meta["id"].astype(int).tolist(),
            "label": val_meta["class"].astype(int).tolist(),
            "pred_061_argmax": val_argmax.tolist(),
            "pred_061": val_open.tolist(),
            "score_061_base": val_base.tolist(),
            "score_061_neighbor": val_neighbor.tolist(),
            "score_061_final": val_final.tolist(),
        }
    )
    test_table = pd.DataFrame(
        {
            "id": test_ids.tolist(),
            "pred_061_argmax": test_argmax.tolist(),
            "pred_061": test_open.tolist(),
            "score_061_base": test_base.tolist(),
            "score_061_neighbor": test_neighbor.tolist(),
            "score_061_final": test_final.tolist(),
        }
    )

    metrics = {
        "val_accuracy_061": float((val_open == val_labels).float().mean().item()),
        "threshold_061": threshold,
        "top_k_061": top_k,
        "base_weight_061": base_weight,
    }
    return val_table, test_table, metrics


def _prepare_insightface_app(config: dict[str, Any]):
    try:
        from insightface.app import FaceAnalysis
        import cv2
    except ImportError as exc:
        raise ImportError("需要在 gpu_env 中安装 insightface 和 onnxruntime-gpu。") from exc

    inf = config["inference"]
    providers = inf.get("onnx_providers")
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name=inf.get("insightface_name", "buffalo_l"), providers=providers)
    app.prepare(ctx_id=int(inf.get("ctx_id", 0)), det_size=tuple(inf.get("det_size", [640, 640])))
    return app, app.models["recognition"], cv2


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return (v / n).astype(np.float32)


def _build_embedder(config: dict[str, Any]):
    app, rec_model, cv2 = _prepare_insightface_app(config)
    inf = config["inference"]
    embedding_mode = str(inf.get("insightface_embedding", "detect_align_with_crop_fallback")).strip().lower()
    tta_hflip = bool(inf.get("tta_horizontal_flip", False))

    def _embed_direct_arcface(bgr: np.ndarray) -> np.ndarray:
        img112 = cv2.resize(bgr, (112, 112))
        raw = rec_model.get_feat(img112)
        return _l2_normalize(raw)

    def _embed_detect_align(bgr: np.ndarray) -> np.ndarray | None:
        faces = app.get(bgr)
        if not faces:
            return None
        return np.asarray(faces[0].normed_embedding, dtype=np.float32).ravel()

    def _embed_single(bgr: np.ndarray, mode: str) -> np.ndarray | None:
        if mode == "detect_align":
            return _embed_detect_align(bgr)
        return _embed_direct_arcface(bgr)

    def _embed_with_tta(bgr: np.ndarray, mode: str) -> np.ndarray | None:
        emb = _embed_single(bgr, mode)
        if emb is None:
            return None
        if tta_hflip:
            bgr_f = cv2.flip(bgr, 1)
            emb2 = _embed_single(bgr_f, mode)
            if emb2 is not None:
                emb = _l2_normalize(emb + emb2)
        return emb

    fallback_count = [0]

    def embed_row(row: pd.Series) -> np.ndarray | None:
        if embedding_mode == "detect_align_with_crop_fallback":
            source_path = row.get("source_path", "")
            if source_path and str(source_path).endswith(".npy"):
                sp = project_path(str(source_path))
                if sp.is_file():
                    arr = np.load(str(sp))
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    emb = _embed_with_tta(bgr, "detect_align")
                    if emb is not None:
                        return emb
            fallback_count[0] += 1
            bgr = _read_bgr(project_path(row["image_path"]))
            if bgr is None:
                return None
            return _embed_with_tta(bgr, "direct_arcface")

        bgr = _read_bgr(project_path(row["image_path"]))
        if bgr is None:
            return None
        return _embed_with_tta(bgr, embedding_mode)

    return embed_row, fallback_count


def _collect_insightface_embeddings(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray], int]:
    splits_dir = project_path(config["data"]["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    embed_row, fallback_count = _build_embedder(config)

    id_to_emb: dict[int, np.ndarray] = {}
    for frame in (train_df, val_df, test_df):
        for _, row in frame.iterrows():
            sid = int(row["id"])
            if sid in id_to_emb:
                continue
            emb = embed_row(row)
            if emb is not None:
                id_to_emb[sid] = emb
    return train_df, val_df, test_df, id_to_emb, int(fallback_count[0])


def _stack_identity(frame: pd.DataFrame, cls: int, id_to_emb: dict[int, np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _, row in frame.iterrows():
        if int(row["class"]) != cls:
            continue
        sid = int(row["id"])
        if sid in id_to_emb:
            out.append(id_to_emb[sid])
    return out


def _calibrate_dual_verifier(
    gallery_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    look_df: pd.DataFrame,
    id_to_emb: dict[int, np.ndarray],
    top_k: int,
    jesse_label: int,
    mila_label: int,
    other_label: int,
    michael_cluster: int,
    sarah_cluster: int,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any], dict[str, Any]]:
    jesse_all = _stack_identity(gallery_df, jesse_label, id_to_emb)
    mila_all = _stack_identity(gallery_df, mila_label, id_to_emb)
    emb_dim = int(next(iter(id_to_emb.values())).shape[0])
    jesse_gallery = np.stack(jesse_all, axis=0) if jesse_all else np.zeros((0, emb_dim), dtype=np.float32)
    mila_gallery = np.stack(mila_all, axis=0) if mila_all else np.zeros((0, emb_dim), dtype=np.float32)

    calib_other_ids = calib_df[calib_df["class"] == other_label]["id"].astype(int).tolist()
    michael_ids = look_df[look_df["cluster"] == michael_cluster]["id"].astype(int).tolist()
    sarah_ids = look_df[look_df["cluster"] == sarah_cluster]["id"].astype(int).tolist()
    michael_ids = [sid for sid in michael_ids if sid in calib_other_ids]
    sarah_ids = [sid for sid in sarah_ids if sid in calib_other_ids]

    fallback_neg = [id_to_emb[sid] for sid in calib_other_ids if sid in id_to_emb]
    michael_embs = [id_to_emb[sid] for sid in michael_ids if sid in id_to_emb] or list(fallback_neg)
    sarah_embs = [id_to_emb[sid] for sid in sarah_ids if sid in id_to_emb] or list(fallback_neg)

    scores_j_pos = np.array(
        [loo_top_k_mean_for_identity(i, jesse_all, top_k) for i in range(len(jesse_all))],
        dtype=np.float64,
    )
    scores_j_neg = np.array(
        [top_k_mean_cosine_similarity(e, jesse_gallery, top_k) for e in michael_embs],
        dtype=np.float64,
    )
    theta_jesse, meta_j = youden_threshold(scores_j_pos, scores_j_neg)

    scores_m_pos = np.array(
        [loo_top_k_mean_for_identity(i, mila_all, top_k) for i in range(len(mila_all))],
        dtype=np.float64,
    )
    scores_m_neg = np.array(
        [top_k_mean_cosine_similarity(e, mila_gallery, top_k) for e in sarah_embs],
        dtype=np.float64,
    )
    theta_mila, meta_m = youden_threshold(scores_m_pos, scores_m_neg)
    return jesse_gallery, mila_gallery, float(theta_jesse), float(theta_mila), meta_j, meta_m


def _score_dual_verifier_frame(
    frame: pd.DataFrame,
    id_to_emb: dict[int, np.ndarray],
    jesse_gallery: np.ndarray,
    mila_gallery: np.ndarray,
    top_k: int,
    theta_jesse: float,
    theta_mila: float,
    other_label: int,
) -> pd.DataFrame:
    ids = frame["id"].astype(int).tolist()
    labels = frame["class"].astype(int).tolist() if "class" in frame.columns else None
    score_jesse: list[float] = []
    score_mila: list[float] = []
    for sid in ids:
        probe = id_to_emb.get(int(sid))
        if probe is None:
            score_jesse.append(float("-inf"))
            score_mila.append(float("-inf"))
            continue
        score_jesse.append(top_k_mean_cosine_similarity(probe, jesse_gallery, top_k))
        score_mila.append(top_k_mean_cosine_similarity(probe, mila_gallery, top_k))

    summary = summarize_dual_verifier_scores(
        score_jesse=np.asarray(score_jesse, dtype=np.float32),
        score_mila=np.asarray(score_mila, dtype=np.float32),
        theta_jesse=theta_jesse,
        theta_mila=theta_mila,
    )
    out = pd.DataFrame(
        {
            "id": ids,
            "pred_067": summary["predictions"].tolist(),
            "score_067_jesse": score_jesse,
            "score_067_mila": score_mila,
            "score_067_selected_excess": summary["selected_excess"].tolist(),
            "score_067_best_excess": summary["best_excess"].tolist(),
            "score_067_excess_jesse": summary["excess_jesse"].tolist(),
            "score_067_excess_mila": summary["excess_mila"].tolist(),
        }
    )
    if labels is not None:
        out.insert(1, "label", labels)
    out.loc[np.isneginf(out["score_067_jesse"]), "pred_067"] = other_label
    return out


def _score_dual_verifier_tables(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_df, val_df, test_df, id_to_emb, fallback_count = _collect_insightface_embeddings(config)
    inf = config["inference"]
    top_k = int(inf.get("gallery_top_k", 5))
    jesse_label = int(inf.get("jesse_class", 1))
    mila_label = int(inf.get("mila_class", 2))
    other_label = int(inf.get("other_class", 0))
    michael_cluster = int(inf.get("michael_like_cluster", 1))
    sarah_cluster = int(inf.get("sarah_like_cluster", 0))
    look_df = _load_lookalike_clusters(project_path(inf["lookalike_assignments_csv"]))

    val_jg, val_mg, theta_j_val, theta_m_val, meta_j_val, meta_m_val = _calibrate_dual_verifier(
        gallery_df=train_df,
        calib_df=train_df,
        look_df=look_df,
        id_to_emb=id_to_emb,
        top_k=top_k,
        jesse_label=jesse_label,
        mila_label=mila_label,
        other_label=other_label,
        michael_cluster=michael_cluster,
        sarah_cluster=sarah_cluster,
    )
    val_table = _score_dual_verifier_frame(
        frame=val_df,
        id_to_emb=id_to_emb,
        jesse_gallery=val_jg,
        mila_gallery=val_mg,
        top_k=top_k,
        theta_jesse=theta_j_val,
        theta_mila=theta_m_val,
        other_label=other_label,
    )

    all_labeled = pd.concat([train_df, val_df], ignore_index=True)
    test_jg, test_mg, theta_j_test, theta_m_test, meta_j_test, meta_m_test = _calibrate_dual_verifier(
        gallery_df=all_labeled,
        calib_df=all_labeled,
        look_df=look_df,
        id_to_emb=id_to_emb,
        top_k=top_k,
        jesse_label=jesse_label,
        mila_label=mila_label,
        other_label=other_label,
        michael_cluster=michael_cluster,
        sarah_cluster=sarah_cluster,
    )
    test_table = _score_dual_verifier_frame(
        frame=test_df,
        id_to_emb=id_to_emb,
        jesse_gallery=test_jg,
        mila_gallery=test_mg,
        top_k=top_k,
        theta_jesse=theta_j_test,
        theta_mila=theta_m_test,
        other_label=other_label,
    )

    val_acc = float((val_table["pred_067"].to_numpy(dtype=np.int64) == val_table["label"].to_numpy(dtype=np.int64)).mean())
    metrics = {
        "gallery_top_k_067": top_k,
        "detect_align_fallback_count": fallback_count,
        "val_theta_jesse": theta_j_val,
        "val_theta_mila": theta_m_val,
        "test_theta_jesse": theta_j_test,
        "test_theta_mila": theta_m_test,
        "val_accuracy_067_train_gallery": val_acc,
        "val_calib_meta_jesse": meta_j_val,
        "val_calib_meta_mila": meta_m_val,
        "test_calib_meta_jesse": meta_j_test,
        "test_calib_meta_mila": meta_m_test,
    }
    return val_table, test_table, metrics


def _load_reference_predictions(path_str: str) -> pd.Series:
    df = pd.read_csv(project_path(path_str))
    return df.set_index("id")["class"].astype(int)


def _prediction_diff_summary(reference: pd.Series, candidate: pd.Series) -> dict[str, Any]:
    aligned = pd.concat([reference.rename("ref"), candidate.rename("cand")], axis=1)
    aligned = aligned.dropna()
    diff = aligned[aligned["ref"] != aligned["cand"]]
    transitions = (
        diff.assign(change=diff["ref"].astype(str) + "->" + diff["cand"].astype(str))["change"].value_counts().to_dict()
    )
    return {
        "num_diff": int(len(diff)),
        "transitions": {str(k): int(v) for k, v in transitions.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="exp_061 + exp_067 score-level gated fusion")
    parser.add_argument("--experiment-name", default="exp_082_vit061_buffalol067_gated_score_fusion_valgrid")
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml",
    )
    parser.add_argument("--primary-checkpoint", default="")
    parser.add_argument(
        "--secondary-config",
        default="configs/experiments/exp_067_buffalo_l_detect_align_dual_verifier_lookalike.yaml",
    )
    parser.add_argument(
        "--reference-primary-submission",
        default="data/submissions/20260402_224630_exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_submission.csv",
    )
    parser.add_argument(
        "--reference-secondary-submission",
        default="data/submissions/20260403_134351_exp_067_buffalo_l_detect_align_dual_verifier_lookalike_submission.csv",
    )
    parser.add_argument("--low-conf-threshold-values", nargs="+", type=float, default=[0.50, 0.525, 0.55, 0.575, 0.60])
    parser.add_argument("--rescue-margin-values", nargs="+", type=float, default=[0.03, 0.05, 0.08, 0.10])
    parser.add_argument("--swap-margin-values", nargs="+", type=float, default=[0.05, 0.08, 0.10, 0.12, 0.15])
    parser.add_argument("--veto-excess-threshold-values", nargs="+", type=float, default=[-0.05, -0.02, 0.0, 0.02])
    args = parser.parse_args()

    exp_name = args.experiment_name
    output_root = ensure_dir(project_path("outputs", exp_name))

    primary_config = load_experiment_config(args.primary_config)
    secondary_config = load_experiment_config(args.secondary_config)
    primary_checkpoint = _resolve_checkpoint_path(primary_config, args.primary_checkpoint or None)

    val_061, test_061, metrics_061 = _score_neighborhood_tables(primary_config, primary_checkpoint)
    val_067, test_067, metrics_067 = _score_dual_verifier_tables(secondary_config)

    val_merged = val_061.merge(val_067, on=["id", "label"], how="inner", validate="one_to_one")
    test_merged = test_061.merge(test_067, on="id", how="inner", validate="one_to_one")

    best_params, search_records = select_best_gated_fusion_params(
        labels=val_merged["label"].to_numpy(dtype=np.int64),
        anchor_predictions=val_merged["pred_061"].to_numpy(dtype=np.int64),
        anchor_scores=val_merged["score_061_final"].to_numpy(dtype=np.float32),
        secondary_predictions=val_merged["pred_067"].to_numpy(dtype=np.int64),
        secondary_selected_excess=val_merged["score_067_selected_excess"].to_numpy(dtype=np.float32),
        secondary_best_excess=val_merged["score_067_best_excess"].to_numpy(dtype=np.float32),
        other_label=0,
        low_conf_threshold_values=args.low_conf_threshold_values,
        rescue_margin_values=args.rescue_margin_values,
        swap_margin_values=args.swap_margin_values,
        veto_excess_threshold_values=args.veto_excess_threshold_values,
    )

    val_fused = apply_gated_anchor_fusion(
        anchor_predictions=val_merged["pred_061"].to_numpy(dtype=np.int64),
        anchor_scores=val_merged["score_061_final"].to_numpy(dtype=np.float32),
        secondary_predictions=val_merged["pred_067"].to_numpy(dtype=np.int64),
        secondary_selected_excess=val_merged["score_067_selected_excess"].to_numpy(dtype=np.float32),
        secondary_best_excess=val_merged["score_067_best_excess"].to_numpy(dtype=np.float32),
        other_label=0,
        low_conf_threshold=float(best_params["low_conf_threshold"]),
        rescue_margin=float(best_params["rescue_margin"]),
        swap_margin=float(best_params["swap_margin"]),
        veto_excess_threshold=float(best_params["veto_excess_threshold"]),
    )
    test_fused = apply_gated_anchor_fusion(
        anchor_predictions=test_merged["pred_061"].to_numpy(dtype=np.int64),
        anchor_scores=test_merged["score_061_final"].to_numpy(dtype=np.float32),
        secondary_predictions=test_merged["pred_067"].to_numpy(dtype=np.int64),
        secondary_selected_excess=test_merged["score_067_selected_excess"].to_numpy(dtype=np.float32),
        secondary_best_excess=test_merged["score_067_best_excess"].to_numpy(dtype=np.float32),
        other_label=0,
        low_conf_threshold=float(best_params["low_conf_threshold"]),
        rescue_margin=float(best_params["rescue_margin"]),
        swap_margin=float(best_params["swap_margin"]),
        veto_excess_threshold=float(best_params["veto_excess_threshold"]),
    )

    val_merged["pred_fused"] = val_fused
    test_merged["pred_fused"] = test_fused

    reference_061 = _load_reference_predictions(args.reference_primary_submission)
    reference_067 = _load_reference_predictions(args.reference_secondary_submission)
    current_061 = test_merged.set_index("id")["pred_061"].astype(int)
    current_067 = test_merged.set_index("id")["pred_067"].astype(int)
    current_fused = test_merged.set_index("id")["pred_fused"].astype(int)

    repro_061 = _prediction_diff_summary(reference_061, current_061)
    repro_067 = _prediction_diff_summary(reference_067, current_067)
    diff_vs_061 = _prediction_diff_summary(reference_061, current_fused)
    diff_vs_067 = _prediction_diff_summary(reference_067, current_fused)

    (output_root / "val_scores.csv").write_text(val_merged.to_csv(index=False), encoding="utf-8")
    (output_root / "test_scores.csv").write_text(test_merged.to_csv(index=False), encoding="utf-8")
    (output_root / "grid_search.json").write_text(json.dumps(search_records, indent=2), encoding="utf-8")

    submission = build_submission_dataframe(pd.read_csv(project_path(primary_config["data"]["splits_dir"]) / "test.csv"), test_fused.tolist())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_path(primary_config["data"]["submissions_dir"], f"{timestamp}_{exp_name}_submission.csv")
    save_submission_dataframe(submission, submission_path)

    metrics = {
        "mode": "gated_score_fusion_061_067",
        "primary_experiment": primary_config["experiment_name"],
        "secondary_experiment": secondary_config["experiment_name"],
        "primary_checkpoint": str(primary_checkpoint),
        **metrics_061,
        **metrics_067,
        "best_params": best_params,
        "val_accuracy_fused": float((val_fused == val_merged["label"].to_numpy(dtype=np.int64)).mean()),
        "num_val_changed_vs_061": int((val_fused != val_merged["pred_061"].to_numpy(dtype=np.int64)).sum()),
        "num_test_changed_vs_061": int((test_fused != test_merged["pred_061"].to_numpy(dtype=np.int64)).sum()),
        "num_test_changed_vs_067": int((test_fused != test_merged["pred_067"].to_numpy(dtype=np.int64)).sum()),
        "reproduction_diff_vs_reference_061": repro_061,
        "reproduction_diff_vs_reference_067": repro_067,
        "fused_diff_vs_reference_061": diff_vs_061,
        "fused_diff_vs_reference_067": diff_vs_067,
        "search_space_size": len(search_records),
        "submission_path": str(submission_path),
    }
    (output_root / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": exp_name,
            "stage": "predict",
            "notes": "scripts/predict_061_067_gated_fusion.py::val_grid",
            "submission_path": str(submission_path),
            "checkpoint_path": f"{primary_checkpoint} | insightface_buffalo_l_gated",
            "config_path": f"{args.primary_config} | {args.secondary_config}",
        },
    )

    print(json.dumps(metrics, indent=2))
    print(f"submission 已生成: {submission_path}")


if __name__ == "__main__":
    main()
