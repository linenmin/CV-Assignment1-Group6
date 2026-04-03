"""
InsightFace（如 buffalo_l）冻结 embedding + 双通道人脸验证 + look-alike 分簇阈值校准。

embedding 提取策略（由 inference.insightface_embedding 配置）：
- ``direct_arcface``：输入已是人脸 crop（如 HAAR），resize 到 112×112 后走 ArcFace get_feat。
- ``detect_align``：输入为整图，走 RetinaFace 检测 + 对齐 → normed_embedding；检不出脸返回 None。
- ``detect_align_with_crop_fallback``（推荐）：优先从 source_path 加载原始整图走 detect_align
  以获取正确对齐的 ArcFace embedding；检测失败时回退到 image_path 的 HAAR crop 走 direct_arcface。
  原始整图检测率 ~98%，回退仅覆盖极少数硬样本。

依赖：pip install insightface onnxruntime-gpu（或 onnxruntime）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from dl_pipeline.common.paths import project_path


def _read_bgr(image_path: Path) -> np.ndarray | None:
    if not image_path.is_file():
        return None
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    return bgr


def top_k_mean_cosine_similarity(
    probe: np.ndarray,
    gallery: np.ndarray,
    k: int,
) -> float:
    """probe/gallery 均已 L2 归一化，余弦相似度为点积。"""
    if gallery.shape[0] == 0:
        return float("-inf")
    sims = gallery @ probe
    k = min(k, len(sims))
    top = np.sort(sims)[-k:]
    return float(top.mean())


def loo_top_k_mean_for_identity(
    probe_index: int,
    identity_embeddings: list[np.ndarray],
    k: int,
) -> float:
    """同一身份 gallery 上 leave-one-out 的 top-k 均值相似度。"""
    probe = identity_embeddings[probe_index]
    others = [identity_embeddings[j] for j in range(len(identity_embeddings)) if j != probe_index]
    if not others:
        return float("-inf")
    g = np.stack(others, axis=0)
    return top_k_mean_cosine_similarity(probe, g, k)


def youden_threshold(scores_positive: np.ndarray, scores_negative: np.ndarray) -> tuple[float, dict[str, Any]]:
    """
    在正负样本分数上搜索使 TPR - FPR 最大的阈值（Youden’s J）。
    分数越高越像“正类”（例如越像 Jesse）。
    在唯一分数取值上枚举，避免 sklearn roc_curve 与 thresholds 长度对齐问题。
    """
    if len(scores_positive) == 0 or len(scores_negative) == 0:
        scores = np.concatenate([scores_positive, scores_negative])
        t = float(np.median(scores)) if len(scores) else 0.0
        return t, {"note": "degenerate_roc", "tpr": 0.0, "fpr": 0.0, "youden_j": 0.0}

    candidates = np.unique(np.concatenate([scores_positive, scores_negative]))
    best_t = float(np.median(candidates))
    best_j = -1.0
    best_tpr = 0.0
    best_fpr = 0.0
    for t in candidates:
        tpr = float((scores_positive >= t).mean())
        fpr = float((scores_negative >= t).mean())
        j = tpr - fpr
        if j > best_j or (j == best_j and t > best_t):
            best_j = j
            best_t = float(t)
            best_tpr = tpr
            best_fpr = fpr
    meta = {
        "tpr_at_best": best_tpr,
        "fpr_at_best": best_fpr,
        "youden_j": best_j,
    }
    return best_t, meta


def dual_verifier_predict(
    score_jesse: float,
    score_mila: float,
    theta_jesse: float,
    theta_mila: float,
) -> int:
    """Jesse=1, Mila=2, other=0。"""
    is_j = score_jesse > theta_jesse
    is_m = score_mila > theta_mila
    if is_j and not is_m:
        return 1
    if is_m and not is_j:
        return 2
    if is_j and is_m:
        return 1 if score_jesse >= score_mila else 2
    return 0


def aggregate_multi_face_scores(
    score_jesse: np.ndarray,
    score_mila: np.ndarray,
    theta_jesse: float,
    theta_mila: float,
) -> dict[str, Any]:
    """
    将同一张图里多张候选脸的 Jesse/Mila 分数聚合为图片级决策。

    规则：
    - Jesse 通道取所有候选脸中的最高 Jesse 分数
    - Mila 通道取所有候选脸中的最高 Mila 分数
    - 再将两条最佳通道分数送入 dual_verifier_predict
    """
    score_jesse = np.asarray(score_jesse, dtype=np.float32).reshape(-1)
    score_mila = np.asarray(score_mila, dtype=np.float32).reshape(-1)
    if score_jesse.shape != score_mila.shape:
        raise ValueError("score_jesse 与 score_mila 形状必须一致。")
    if score_jesse.size == 0:
        return {
            "prediction": 0,
            "best_score_jesse": float("-inf"),
            "best_score_mila": float("-inf"),
            "best_excess_jesse": float("-inf"),
            "best_excess_mila": float("-inf"),
            "best_face_index_jesse": -1,
            "best_face_index_mila": -1,
        }

    idx_jesse = int(np.argmax(score_jesse))
    idx_mila = int(np.argmax(score_mila))
    best_score_jesse = float(score_jesse[idx_jesse])
    best_score_mila = float(score_mila[idx_mila])
    best_excess_jesse = best_score_jesse - float(theta_jesse)
    best_excess_mila = best_score_mila - float(theta_mila)
    prediction = dual_verifier_predict(best_score_jesse, best_score_mila, theta_jesse, theta_mila)
    return {
        "prediction": int(prediction),
        "best_score_jesse": best_score_jesse,
        "best_score_mila": best_score_mila,
        "best_excess_jesse": float(best_excess_jesse),
        "best_excess_mila": float(best_excess_mila),
        "best_face_index_jesse": idx_jesse,
        "best_face_index_mila": idx_mila,
    }


def describe_multi_face_scores(
    score_jesse: np.ndarray,
    score_mila: np.ndarray,
    theta_jesse: float,
    theta_mila: float,
) -> list[dict[str, Any]]:
    """
    返回每张候选脸的通道分数明细，便于导出可视化或审计表。
    """
    score_jesse = np.asarray(score_jesse, dtype=np.float32).reshape(-1)
    score_mila = np.asarray(score_mila, dtype=np.float32).reshape(-1)
    if score_jesse.shape != score_mila.shape:
        raise ValueError("score_jesse 与 score_mila 形状必须一致。")
    if score_jesse.size == 0:
        return []

    idx_jesse = int(np.argmax(score_jesse))
    idx_mila = int(np.argmax(score_mila))
    rows: list[dict[str, Any]] = []
    for i in range(score_jesse.size):
        sj = float(score_jesse[i])
        sm = float(score_mila[i])
        rows.append(
            {
                "face_index": int(i),
                "score_jesse": sj,
                "score_mila": sm,
                "excess_jesse": sj - float(theta_jesse),
                "excess_mila": sm - float(theta_mila),
                "passes_jesse": bool(sj > theta_jesse),
                "passes_mila": bool(sm > theta_mila),
                "is_best_jesse": bool(i == idx_jesse),
                "is_best_mila": bool(i == idx_mila),
            }
        )
    return rows


def _load_lookalike_clusters(lookalike_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(lookalike_csv)
    need = {"id", "split", "cluster"}
    if not need.issubset(df.columns):
        raise ValueError(f"look-alike CSV 需含列: {need}，当前: {df.columns.tolist()}")
    return df


def run_insightface_dual_verifier(config: dict[str, Any], test_df: pd.DataFrame) -> list[int]:
    """
    全流程：提取 embedding → 双通道阈值（Jesse vs Michael-like, Mila vs Sarah-like）→ 测试集预测。
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise ImportError(
            "exp_066 需要安装 insightface 与 onnxruntime："
            "pip install insightface onnxruntime-gpu"
        ) from e

    inf = config.get("inference", {})
    data_cfg = config["data"]
    splits_dir = project_path(data_cfg["splits_dir"])
    lookalike_csv = project_path(inf.get("lookalike_assignments_csv", "reports/analysis/exp_029_other_k2/other_cluster_assignments.csv"))

    pack = inf.get("insightface_name", "buffalo_l")
    det_size = tuple(inf.get("det_size", [640, 640]))
    embedding_mode = str(inf.get("insightface_embedding", "detect_align_with_crop_fallback")).strip().lower()
    valid_modes = ("direct_arcface", "detect_align", "detect_align_with_crop_fallback")
    if embedding_mode not in valid_modes:
        raise ValueError(
            f"inference.insightface_embedding 须为 {valid_modes} 之一，"
            f"当前: {embedding_mode!r}"
        )
    ctx_id = int(inf.get("ctx_id", 0))
    top_k = int(inf.get("gallery_top_k", 5))
    tta_hflip = bool(inf.get("tta_horizontal_flip", False))
    jesse_label = int(inf.get("jesse_class", 1))
    mila_label = int(inf.get("mila_class", 2))
    other_label = int(inf.get("other_class", 0))
    # exp_029 聚类：cluster 1 更靠近 Jesse（Michael-like），cluster 0 更靠近 Mila（Sarah-like）
    michael_cluster = int(inf.get("michael_like_cluster", 1))
    sarah_cluster = int(inf.get("sarah_like_cluster", 0))

    providers = inf.get("onnx_providers")
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    app = FaceAnalysis(name=pack, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    rec_model = app.models["recognition"]

    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).ravel()
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return v
        return (v / n).astype(np.float32)

    def _embed_direct_arcface(bgr: np.ndarray) -> np.ndarray:
        """输入已是人脸区域：单一确定路径，resize → ArcFace，不做检测。"""
        img112 = cv2.resize(bgr, (112, 112))
        raw = rec_model.get_feat(img112)
        return _l2_normalize(raw)

    def _embed_detect_align(bgr: np.ndarray) -> np.ndarray | None:
        """整图：仅用 RetinaFace 检测对齐后的 normed_embedding；无脸则 None（不回退）。"""
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
                emb = emb + emb2
                emb = _l2_normalize(emb)
        return emb

    fallback_count = [0]  # 用于统计回退次数

    def embed_row(row: pd.Series) -> np.ndarray | None:
        if embedding_mode == "detect_align_with_crop_fallback":
            # 优先：从原始整图走 detect_align（正确对齐的 ArcFace 输入）
            source_path = row.get("source_path", "")
            if source_path and str(source_path).endswith(".npy"):
                sp = project_path(str(source_path))
                if sp.is_file():
                    arr = np.load(str(sp))
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    emb = _embed_with_tta(bgr, "detect_align")
                    if emb is not None:
                        return emb
            # 回退：HAAR crop → direct_arcface
            fallback_count[0] += 1
            p = project_path(row["image_path"])
            bgr = _read_bgr(p)
            if bgr is None:
                return None
            return _embed_with_tta(bgr, "direct_arcface")
        else:
            p = project_path(row["image_path"])
            bgr = _read_bgr(p)
            if bgr is None:
                return None
            return _embed_with_tta(bgr, embedding_mode)

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    look_df = _load_lookalike_clusters(lookalike_csv)

    id_to_emb: dict[int, np.ndarray] = {}

    def collect_embeddings(frame: pd.DataFrame) -> None:
        for _, row in frame.iterrows():
            sid = int(row["id"])
            if sid in id_to_emb:
                continue
            e = embed_row(row)
            if e is not None:
                id_to_emb[sid] = e

    collect_embeddings(train_df)
    collect_embeddings(val_df)
    collect_embeddings(test_df)

    def stack_class(df: pd.DataFrame, cls: int) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for _, row in df.iterrows():
            if int(row["class"]) != cls:
                continue
            sid = int(row["id"])
            if sid in id_to_emb:
                out.append(id_to_emb[sid])
        return out

    # frozen model 无训练 → 合并 train+val 全部标注数据用于 gallery 和阈值校准
    all_labeled = pd.concat([train_df, val_df], ignore_index=True)

    jesse_all = stack_class(all_labeled, jesse_label)
    mila_all = stack_class(all_labeled, mila_label)
    emb_dim = 512
    if jesse_all:
        emb_dim = int(jesse_all[0].shape[0])
    elif mila_all:
        emb_dim = int(mila_all[0].shape[0])
    elif id_to_emb:
        emb_dim = int(next(iter(id_to_emb.values())).shape[0])
    jesse_gallery = np.stack(jesse_all, axis=0) if jesse_all else np.zeros((0, emb_dim), dtype=np.float32)
    mila_gallery = np.stack(mila_all, axis=0) if mila_all else np.zeros((0, emb_dim), dtype=np.float32)

    all_ids_class0 = all_labeled[all_labeled["class"] == other_label]["id"].astype(int).tolist()
    # look-alike 聚类不区分 train/val split，使用全部 other 样本
    michael_ids = look_df[look_df["cluster"] == michael_cluster]["id"].astype(int).tolist()
    sarah_ids = look_df[look_df["cluster"] == sarah_cluster]["id"].astype(int).tolist()
    michael_ids = [i for i in michael_ids if i in all_ids_class0]
    sarah_ids = [i for i in sarah_ids if i in all_ids_class0]

    # 回退：若分簇缺失则用全部 other 标注样本作双通道负样本（次优）
    fallback_neg = [id_to_emb[i] for i in all_ids_class0 if i in id_to_emb]
    michael_embs = [id_to_emb[i] for i in michael_ids if i in id_to_emb]
    sarah_embs = [id_to_emb[i] for i in sarah_ids if i in id_to_emb]
    if not michael_embs:
        michael_embs = list(fallback_neg)
    if not sarah_embs:
        sarah_embs = list(fallback_neg)

    # --- 校准 Jesse 通道：正样本 = 全部 Jesse LOO top-k；负样本 = Michael-like vs 全 Jesse gallery ---
    scores_j_pos = np.array(
        [
            loo_top_k_mean_for_identity(i, jesse_all, top_k)
            for i in range(len(jesse_all))
        ],
        dtype=np.float64,
    )
    scores_j_neg = np.array(
        [top_k_mean_cosine_similarity(e, jesse_gallery, top_k) for e in michael_embs],
        dtype=np.float64,
    )
    theta_jesse, meta_j = youden_threshold(scores_j_pos, scores_j_neg)

    # --- 校准 Mila 通道 ---
    scores_m_pos = np.array(
        [loo_top_k_mean_for_identity(i, mila_all, top_k) for i in range(len(mila_all))],
        dtype=np.float64,
    )
    scores_m_neg = np.array(
        [top_k_mean_cosine_similarity(e, mila_gallery, top_k) for e in sarah_embs],
        dtype=np.float64,
    )
    theta_mila, meta_m = youden_threshold(scores_m_pos, scores_m_neg)

    # --- 验证集准确率（仅日志）---
    val_acc = None
    if len(val_df) and "class" in val_df.columns:
        correct = 0
        total = 0
        for _, row in val_df.iterrows():
            sid = int(row["id"])
            if sid not in id_to_emb:
                continue
            probe = id_to_emb[sid]
            sj = top_k_mean_cosine_similarity(probe, jesse_gallery, top_k)
            sm = top_k_mean_cosine_similarity(probe, mila_gallery, top_k)
            pred = dual_verifier_predict(sj, sm, theta_jesse, theta_mila)
            if int(pred) == int(row["class"]):
                correct += 1
            total += 1
        val_acc = correct / total if total else None

    # --- 测试集 ---
    predictions: list[int] = []
    for _, row in test_df.iterrows():
        sid = int(row["id"])
        if sid not in id_to_emb:
            predictions.append(other_label)
            continue
        probe = id_to_emb[sid]
        sj = top_k_mean_cosine_similarity(probe, jesse_gallery, top_k)
        sm = top_k_mean_cosine_similarity(probe, mila_gallery, top_k)
        predictions.append(dual_verifier_predict(sj, sm, theta_jesse, theta_mila))

    output_root = project_path("outputs", config["experiment_name"])
    output_root.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mode": "insightface_dual_verifier",
        "insightface_pack": pack,
        "insightface_embedding": embedding_mode,
        "detect_align_fallback_count": fallback_count[0],
        "gallery_top_k": top_k,
        "tta_horizontal_flip": tta_hflip,
        "theta_jesse": theta_jesse,
        "theta_mila": theta_mila,
        "calib_meta_jesse": meta_j,
        "calib_meta_mila": meta_m,
        "num_jesse_gallery": len(jesse_all),
        "num_mila_gallery": len(mila_all),
        "num_michael_calib": len(michael_embs),
        "num_sarah_calib": len(sarah_embs),
        "val_accuracy_dual_verifier": val_acc,
        "lookalike_csv": str(lookalike_csv),
    }
    (output_root / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return predictions


def run_insightface_dual_verifier_from_config(config: dict[str, Any]) -> tuple[list[int], pd.DataFrame]:
    splits_dir = project_path(config["data"]["splits_dir"])
    test_df = pd.read_csv(splits_dir / "test.csv")
    preds = run_insightface_dual_verifier(config, test_df)
    return preds, test_df
