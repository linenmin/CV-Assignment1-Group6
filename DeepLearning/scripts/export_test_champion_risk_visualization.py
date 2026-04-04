from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import _collect_embeddings, _load_model_from_checkpoint
from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.inference.insightface_dual_verifier import (
    _load_lookalike_clusters,
    aggregate_multi_face_scores,
    describe_multi_face_scores,
    loo_top_k_mean_for_identity,
    top_k_mean_cosine_similarity,
    youden_threshold,
)
from dl_pipeline.inference.prototype import (
    combine_embedding_sets,
    compute_class_prototypes,
    compute_similarity_matrix,
    neighborhood_aware_predictions,
)


PANEL_W = 1480
PANEL_H = 1060
HEADER_H = 84
GAP = 16
RAW_BOX_W = 760
FACE_THUMB = 112
TEXT = (25, 25, 25)
BG = (248, 248, 248)
BOX_DEFAULT = (120, 120, 120)
BOX_JESSE = (40, 120, 255)
BOX_MILA = (255, 120, 40)
BOX_BOTH = (30, 180, 70)
STRONG_EXPERIMENTS = {
    "e061": "exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta",
    "e067": "exp_067_buffalo_l_detect_align_dual_verifier_lookalike",
    "e082": "exp_082_vit061_buffalol067_gated_score_fusion_valgrid",
    "e086": "exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike",
    "e087": "exp_087_multiface_exact_raw_hash_override",
    "e088": "exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta",
}


def _load_raw_bgr(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path), allow_pickle=False)
    return arr


def _load_raw_rgb(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path), allow_pickle=False)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _read_bgr(image_path: Path) -> np.ndarray | None:
    if not image_path.is_file():
        return None
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def _load_png_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fit_rgb(image_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _prepare_insightface_app(config: dict[str, Any]):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError("need insightface and onnxruntime-gpu in gpu_env") from exc

    inf = config["inference"]
    providers = inf.get("onnx_providers")
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name=inf.get("insightface_name", "buffalo_l"), providers=providers)
    app.prepare(ctx_id=int(inf.get("ctx_id", 0)), det_size=tuple(inf.get("det_size", [640, 640])))
    return app, app.models["recognition"]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return (v / n).astype(np.float32)


def _embed_direct_arcface(rec_model, bgr: np.ndarray) -> np.ndarray:
    img112 = cv2.resize(bgr, (112, 112))
    raw = rec_model.get_feat(img112)
    return _l2_normalize(raw)


def _extract_face_candidates(app, bgr: np.ndarray) -> list[dict[str, Any]]:
    faces = app.get(bgr)
    out: list[dict[str, Any]] = []
    for idx, face in enumerate(faces):
        bbox = np.asarray(face.bbox, dtype=np.float32).tolist()
        out.append(
            {
                "face_index": idx,
                "bbox": [float(v) for v in bbox],
                "embedding": np.asarray(face.normed_embedding, dtype=np.float32).ravel(),
            }
        )
    return out


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
):
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

    scores_j_pos = np.array([loo_top_k_mean_for_identity(i, jesse_all, top_k) for i in range(len(jesse_all))], dtype=np.float64)
    scores_j_neg = np.array([top_k_mean_cosine_similarity(e, jesse_gallery, top_k) for e in michael_embs], dtype=np.float64)
    theta_jesse, _ = youden_threshold(scores_j_pos, scores_j_neg)

    scores_m_pos = np.array([loo_top_k_mean_for_identity(i, mila_all, top_k) for i in range(len(mila_all))], dtype=np.float64)
    scores_m_neg = np.array([top_k_mean_cosine_similarity(e, mila_gallery, top_k) for e in sarah_embs], dtype=np.float64)
    theta_mila, _ = youden_threshold(scores_m_pos, scores_m_neg)

    return jesse_gallery, mila_gallery, float(theta_jesse), float(theta_mila)


def _collect_single_embeddings(config: dict[str, Any], app, rec_model):
    inf = config["inference"]
    embedding_mode = str(inf.get("insightface_embedding", "detect_align_with_crop_fallback")).strip().lower()
    splits_dir = project_path(config["data"]["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    id_to_emb: dict[int, np.ndarray] = {}
    for frame in (train_df, val_df):
        for _, row in frame.iterrows():
            sid = int(row["id"])
            if sid in id_to_emb:
                continue
            emb = None
            if embedding_mode == "detect_align_with_crop_fallback":
                source_path = row.get("source_path", "")
                if source_path and str(source_path).endswith(".npy"):
                    sp = project_path(str(source_path))
                    if sp.is_file():
                        bgr = _load_raw_bgr(sp)
                        faces = _extract_face_candidates(app, bgr)
                        if faces:
                            emb = faces[0]["embedding"]
                if emb is None:
                    crop_bgr = _read_bgr(project_path(str(row["image_path"])))
                    if crop_bgr is not None:
                        emb = _embed_direct_arcface(rec_model, crop_bgr)
            else:
                crop_bgr = _read_bgr(project_path(str(row["image_path"])))
                if crop_bgr is not None:
                    emb = _embed_direct_arcface(rec_model, crop_bgr)
            if emb is not None:
                id_to_emb[sid] = emb
    return train_df, val_df, id_to_emb


def _bbox_to_ints(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(width, int(round(x2))))
    iy2 = max(iy1 + 1, min(height, int(round(y2))))
    return ix1, iy1, ix2, iy2


def _make_missing_panel(message: str, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 230, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (160, 160, 160), 2)
    y = height // 2 - 12
    for line in message.split("\n"):
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
        x = max(8, (width - size[0]) // 2)
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 1, cv2.LINE_AA)
        y += 28
    return panel


def _draw_header(canvas: np.ndarray, text: str) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], HEADER_H), (235, 235, 235), -1)
    cv2.putText(canvas, text, (GAP, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.82, TEXT, 2, cv2.LINE_AA)


def _risk_from_threshold_margin(margin: float) -> float:
    return max(0.0, 1.0 - min(1.0, margin / 0.12))


def _risk_from_class_margin(margin: float) -> float:
    return max(0.0, 1.0 - min(1.0, margin / 0.10))


def _load_submission_predictions(paths: dict[str, str]) -> pd.DataFrame:
    merged = None
    for alias, rel_path in paths.items():
        frame = pd.read_csv(project_path(rel_path))[["id", "class"]].rename(columns={"class": alias})
        merged = frame if merged is None else merged.merge(frame, on="id", how="inner")
    if merged is None:
        raise FileNotFoundError("no strong submission paths")
    return merged.sort_values("id").reset_index(drop=True)


def _render_sample_canvas(
    raw_rgb: np.ndarray | None,
    haar_rgb: np.ndarray | None,
    mtcnn_rgb: np.ndarray | None,
    selected_rgb: np.ndarray | None,
    face_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    row: pd.Series,
) -> np.ndarray:
    canvas = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    header = (
        f"rank={int(row['rank'])} | id={int(row['id'])} | risk={float(row['risk_score']):.3f} "
        f"| pred={int(row['pred_088'])} | multiface={int(row['multiface_pred'])} | source={row['probe_source']}"
    )
    _draw_header(canvas, header)

    raw_panel_h = PANEL_H - HEADER_H - GAP * 2 - FACE_THUMB - 54
    if raw_rgb is None:
        raw_panel = _make_missing_panel("raw\nmissing", RAW_BOX_W, raw_panel_h)
    else:
        annotated = raw_rgb.copy()
        ih, iw = annotated.shape[:2]
        for face_row, score_row in zip(face_rows, score_rows):
            x1, y1, x2, y2 = _bbox_to_ints(face_row["bbox"], iw, ih)
            color = BOX_DEFAULT
            if score_row["is_best_jesse"] and score_row["is_best_mila"]:
                color = BOX_BOTH
            elif score_row["is_best_jesse"]:
                color = BOX_JESSE
            elif score_row["is_best_mila"]:
                color = BOX_MILA
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"F{score_row['face_index']} J:{score_row['score_jesse']:.3f} M:{score_row['score_mila']:.3f}"
            cv2.putText(annotated, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        raw_panel = _fit_rgb(annotated, RAW_BOX_W, raw_panel_h)
    canvas[HEADER_H + GAP:HEADER_H + GAP + raw_panel_h, GAP:GAP + RAW_BOX_W] = raw_panel

    right_x = GAP * 2 + RAW_BOX_W
    right_w = PANEL_W - right_x - GAP
    y = HEADER_H + GAP
    small_h = 145
    crop_panels = [("HAAR", haar_rgb), ("MTCNN", mtcnn_rgb), ("SEL", selected_rgb)]
    for idx, (title, image_rgb) in enumerate(crop_panels):
        cv2.putText(canvas, title, (right_x + idx * (right_w // 3), y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT, 2, cv2.LINE_AA)
        panel = _make_missing_panel(f"{title}\nmissing", right_w // 3 - 8, small_h) if image_rgb is None else _fit_rgb(image_rgb, right_w // 3 - 8, small_h)
        x0 = right_x + idx * (right_w // 3)
        canvas[y + 28:y + 28 + small_h, x0:x0 + panel.shape[1]] = panel
    y += small_h + 54

    lines = [
        f"final={float(row['final_score']):.4f}  threshold={float(row['selected_threshold']):.4f}  |margin|={float(row['threshold_margin_abs']):.4f}",
        f"cls1={float(row['score_class1']):.4f}  cls2={float(row['score_class2']):.4f}  class_margin={float(row['class_margin']):.4f}",
        f"base={float(row['base_score']):.4f}  neighbor={float(row['neighbor_score']):.4f}  strong_support={float(row['strong_support_share']):.3f}",
        f"multiface bestJ={float(row['best_score_jesse']):.4f} ({float(row['best_excess_jesse']):+.4f})  bestM={float(row['best_score_mila']):.4f} ({float(row['best_excess_mila']):+.4f})",
        f"risk parts: th={float(row['risk_threshold']):.3f} cls={float(row['risk_class_margin']):.3f} strong={float(row['risk_strong_disagree']):.3f} multi={float(row['risk_multiface_conflict']):.3f} source={float(row['risk_input_issue']):.3f}",
    ]
    text_y = y + 10
    for line in lines:
        cv2.putText(canvas, line, (right_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT, 1, cv2.LINE_AA)
        text_y += 24

    cv2.putText(canvas, "Strong Models", (right_x, text_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT, 2, cv2.LINE_AA)
    text_y += 36
    strong_line = "  ".join([f"{alias}:{int(row[alias])}" for alias in STRONG_EXPERIMENTS])
    cv2.putText(canvas, strong_line, (right_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)

    strip_y = PANEL_H - GAP - FACE_THUMB
    x = GAP
    if not face_rows:
        panel = _make_missing_panel("no faces", FACE_THUMB, FACE_THUMB)
        canvas[strip_y:strip_y + FACE_THUMB, x:x + FACE_THUMB] = panel
    elif raw_rgb is not None:
        ih, iw = raw_rgb.shape[:2]
        for face_row, score_row in zip(face_rows[:8], score_rows[:8]):
            x1, y1, x2, y2 = _bbox_to_ints(face_row["bbox"], iw, ih)
            crop = raw_rgb[y1:y2, x1:x2]
            thumb = _fit_rgb(crop, FACE_THUMB, FACE_THUMB) if crop.size else _make_missing_panel("empty", FACE_THUMB, FACE_THUMB)
            canvas[strip_y:strip_y + FACE_THUMB, x:x + FACE_THUMB] = thumb
            color = BOX_DEFAULT
            if score_row["is_best_jesse"] and score_row["is_best_mila"]:
                color = BOX_BOTH
            elif score_row["is_best_jesse"]:
                color = BOX_JESSE
            elif score_row["is_best_mila"]:
                color = BOX_MILA
            cv2.rectangle(canvas, (x, strip_y), (x + FACE_THUMB - 1, strip_y + FACE_THUMB - 1), color, 3)
            cv2.putText(canvas, f"F{score_row['face_index']}", (x + 4, strip_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            x += FACE_THUMB + 12
            if x + FACE_THUMB > RAW_BOX_W:
                break
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Export champion-risk test sample visualizations for exp_088.")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml",
    )
    parser.add_argument(
        "--multiface-config",
        default="configs/experiments/exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="data/visualizations/test_champion_risk_probe/top50_exp_088",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--ids", nargs="*", type=int, default=[])
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    splits_dir = project_path(config["data"]["splits_dir"])
    metrics = json.loads((project_path("outputs", config["experiment_name"], "metrics.json")).read_text(encoding="utf-8"))
    checkpoint_path = metrics["best_model_path"]

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

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

    model = _load_model_from_checkpoint(config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    inference_config = config.get("inference", {})
    use_horizontal_flip_tta = inference_config.get("tta_horizontal_flip", False)
    other_label = int(inference_config.get("other_label", 0))
    prototype_labels = inference_config.get("prototype_labels", [1, 2])
    threshold = float(inference_config.get("threshold", 0.55))
    neighborhood_config = inference_config.get("neighborhood_aware", {})
    top_k = int(neighborhood_config.get("top_k", 15))
    base_weight = float(neighborhood_config.get("base_weight", 0.5))

    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device, use_horizontal_flip_tta)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device, use_horizontal_flip_tta)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device, use_horizontal_flip_tta)

    all_gallery_embeddings, all_gallery_labels = combine_embedding_sets([(train_embeddings, train_labels), (val_embeddings, val_labels)])
    final_prototypes = compute_class_prototypes(all_gallery_embeddings, all_gallery_labels, prototype_labels=prototype_labels)
    test_predictions, test_base_scores, test_neighbor_scores, test_final_scores = neighborhood_aware_predictions(
        query_embeddings=test_embeddings,
        prototypes=final_prototypes,
        other_label=other_label,
        threshold=threshold,
        top_k=top_k,
        base_weight=base_weight,
    )
    sim_matrix, sim_labels = compute_similarity_matrix(test_embeddings, final_prototypes, prototype_labels=list(prototype_labels))
    score_class1 = sim_matrix[:, sim_labels.index(1)] if 1 in sim_labels else torch.zeros_like(test_final_scores)
    score_class2 = sim_matrix[:, sim_labels.index(2)] if 2 in sim_labels else torch.zeros_like(test_final_scores)
    top2_scores, _ = sim_matrix.topk(k=min(2, sim_matrix.shape[1]), dim=1)
    class_margin = top2_scores[:, 0] - top2_scores[:, 1] if top2_scores.shape[1] > 1 else torch.ones_like(test_final_scores)
    threshold_margin_abs = torch.abs(test_final_scores - threshold)

    champion_df = pd.DataFrame(
        {
            "id": test_ids.detach().cpu().numpy().astype(int),
            "pred_088": test_predictions.detach().cpu().numpy().astype(int),
            "base_score": test_base_scores.detach().cpu().numpy(),
            "neighbor_score": test_neighbor_scores.detach().cpu().numpy(),
            "final_score": test_final_scores.detach().cpu().numpy(),
            "selected_threshold": threshold,
            "threshold_margin_abs": threshold_margin_abs.detach().cpu().numpy(),
            "score_class1": score_class1.detach().cpu().numpy(),
            "score_class2": score_class2.detach().cpu().numpy(),
            "class_margin": class_margin.detach().cpu().numpy(),
        }
    ).sort_values("id").reset_index(drop=True)

    strong_pred_df = _load_submission_predictions(
        {
            alias: f"data/submissions/{sorted(project_path('data', 'submissions').glob(f'*_{exp_name}_submission.csv'))[-1].name}"
            for alias, exp_name in STRONG_EXPERIMENTS.items()
        }
    )
    merged = champion_df.merge(strong_pred_df, on="id", how="left").merge(test_df, on="id", how="left")
    support_counts = []
    for _, row in merged.iterrows():
        champion_pred = int(row["pred_088"])
        support = sum(int(row[alias]) == champion_pred for alias in STRONG_EXPERIMENTS)
        support_counts.append(support / len(STRONG_EXPERIMENTS))
    merged["strong_support_share"] = support_counts

    mf_config = load_experiment_config(args.multiface_config)
    app, rec_model = _prepare_insightface_app(mf_config)
    mf_train_df, mf_val_df, id_to_emb = _collect_single_embeddings(mf_config, app, rec_model)
    labeled_df = pd.concat([mf_train_df, mf_val_df], ignore_index=True)
    inf = mf_config["inference"]
    gallery_top_k = int(inf.get("gallery_top_k", 5))
    jesse_label = int(inf.get("jesse_class", 1))
    mila_label = int(inf.get("mila_class", 2))
    mf_other_label = int(inf.get("other_class", 0))
    michael_cluster = int(inf.get("michael_like_cluster", 1))
    sarah_cluster = int(inf.get("sarah_like_cluster", 0))
    look_df = _load_lookalike_clusters(project_path(inf["lookalike_assignments_csv"]))
    jesse_gallery, mila_gallery, theta_jesse, theta_mila = _calibrate_dual_verifier(
        gallery_df=labeled_df,
        calib_df=labeled_df,
        look_df=look_df,
        id_to_emb=id_to_emb,
        top_k=gallery_top_k,
        jesse_label=jesse_label,
        mila_label=mila_label,
        other_label=mf_other_label,
        michael_cluster=michael_cluster,
        sarah_cluster=sarah_cluster,
    )

    audit_rows = []
    face_details = []
    for _, row in merged.iterrows():
        sid = int(row["id"])
        raw_path = project_path(str(row["source_path"]))
        raw_bgr = _load_raw_bgr(raw_path) if raw_path.is_file() else None
        face_rows = _extract_face_candidates(app, raw_bgr) if raw_bgr is not None else []
        probe_source = "detect_align" if face_rows else "missing"
        if not face_rows:
            fallback_bgr = _read_bgr(project_path(str(row["image_path"])))
            if fallback_bgr is not None:
                h, w = fallback_bgr.shape[:2]
                face_rows = [
                    {
                        "face_index": 0,
                        "bbox": [0.0, 0.0, float(w), float(h)],
                        "embedding": _embed_direct_arcface(rec_model, fallback_bgr),
                    }
                ]
                probe_source = "crop_fallback"

        score_jesse = np.array([top_k_mean_cosine_similarity(face["embedding"], jesse_gallery, gallery_top_k) for face in face_rows], dtype=np.float32)
        score_mila = np.array([top_k_mean_cosine_similarity(face["embedding"], mila_gallery, gallery_top_k) for face in face_rows], dtype=np.float32)
        agg = aggregate_multi_face_scores(score_jesse, score_mila, theta_jesse, theta_mila)
        score_rows = describe_multi_face_scores(score_jesse, score_mila, theta_jesse, theta_mila)

        audit_rows.append(
            {
                "id": sid,
                "probe_source": probe_source,
                "num_faces_detected": int(len(face_rows)),
                "multiface_pred": int(agg["prediction"]),
                "best_score_jesse": float(agg["best_score_jesse"]),
                "best_score_mila": float(agg["best_score_mila"]),
                "best_excess_jesse": float(agg["best_excess_jesse"]),
                "best_excess_mila": float(agg["best_excess_mila"]),
            }
        )
        for face_row, score_row in zip(face_rows, score_rows):
            face_details.append(
                {
                    "id": sid,
                    "bbox_x1": face_row["bbox"][0],
                    "bbox_y1": face_row["bbox"][1],
                    "bbox_x2": face_row["bbox"][2],
                    "bbox_y2": face_row["bbox"][3],
                    **score_row,
                }
            )

    merged = merged.merge(pd.DataFrame(audit_rows), on="id", how="left")
    merged["risk_threshold"] = merged["threshold_margin_abs"].apply(_risk_from_threshold_margin)
    merged["risk_class_margin"] = merged["class_margin"].apply(_risk_from_class_margin)
    merged["risk_strong_disagree"] = 1.0 - merged["strong_support_share"]
    merged["risk_multiface_conflict"] = (
        (merged["pred_088"] != merged["multiface_pred"]).astype(float) * 0.7
        + (1.0 - merged[["best_excess_jesse", "best_excess_mila"]].max(axis=1).clip(lower=0.0, upper=0.25) / 0.25) * 0.3
    )
    merged["risk_input_issue"] = (
        (merged["probe_source"] == "crop_fallback").astype(float) * 0.7
        + (merged["num_faces_detected"].fillna(0).clip(upper=4) / 4.0) * 0.3
    )
    merged["risk_score"] = (
        merged["risk_threshold"] * 0.30
        + merged["risk_class_margin"] * 0.25
        + merged["risk_strong_disagree"] * 0.20
        + merged["risk_multiface_conflict"] * 0.15
        + merged["risk_input_issue"] * 0.10
    )
    merged = merged.sort_values(
        ["risk_score", "risk_threshold", "risk_class_margin", "risk_strong_disagree", "id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    if args.ids:
        merged = merged[merged["id"].isin(set(args.ids))].copy()
        merged = merged.sort_values(["rank", "id"]).reset_index(drop=True)
    else:
        merged = merged.head(int(args.top_n)).copy()

    out_root = ensure_dir(project_path(args.out_dir))
    images_root = ensure_dir(out_root / "images")
    mtcnn_dir = project_path("data", "processed", "exp_008_mtcnn_faces_112", "test")

    for _, row in merged.iterrows():
        sid = int(row["id"])
        raw_path = project_path(str(row["source_path"]))
        raw_rgb = _load_raw_rgb(raw_path) if raw_path.is_file() else None
        raw_bgr = _load_raw_bgr(raw_path) if raw_path.is_file() else None
        haar_rgb = _load_png_rgb(project_path(str(row["image_path"])))
        mtcnn_rgb = _load_png_rgb(mtcnn_dir / f"{sid}.png")
        selected_rgb = _load_png_rgb(project_path(str(row["image_path"])))

        face_rows = _extract_face_candidates(app, raw_bgr) if raw_bgr is not None else []
        if not face_rows:
            fallback_bgr = _read_bgr(project_path(str(row["image_path"])))
            probe_source = "missing"
            if fallback_bgr is not None:
                h, w = fallback_bgr.shape[:2]
                face_rows = [
                    {
                        "face_index": 0,
                        "bbox": [0.0, 0.0, float(w), float(h)],
                        "embedding": _embed_direct_arcface(rec_model, fallback_bgr),
                    }
                ]
                probe_source = "crop_fallback"
        else:
            probe_source = "detect_align"
        score_jesse = np.array([top_k_mean_cosine_similarity(face["embedding"], jesse_gallery, gallery_top_k) for face in face_rows], dtype=np.float32)
        score_mila = np.array([top_k_mean_cosine_similarity(face["embedding"], mila_gallery, gallery_top_k) for face in face_rows], dtype=np.float32)
        score_rows = describe_multi_face_scores(score_jesse, score_mila, theta_jesse, theta_mila)

        row = row.copy()
        row["probe_source"] = probe_source
        canvas = _render_sample_canvas(raw_rgb, haar_rgb, mtcnn_rgb, selected_rgb, face_rows, score_rows, row)
        stem = f"rank_{int(row['rank']):03d}_id{sid:04d}_pred{int(row['pred_088'])}_risk{float(row['risk_score']):.3f}"
        Image.fromarray(canvas).save(images_root / f"{stem}.png", format="PNG", compress_level=3)

    merged.to_csv(out_root / "summary.csv", index=False)
    pd.DataFrame(face_details).to_csv(out_root / "face_details.csv", index=False)
    pd.DataFrame([{"alias": alias, "experiment_name": exp} for alias, exp in STRONG_EXPERIMENTS.items()]).to_csv(out_root / "strong_experiments.csv", index=False)
    (out_root / "README.md").write_text(
        "# champion risk probe\n\n"
        "- rank by exp_088 internal risk, not by all-submission disagreement\n"
        "- risk combines threshold margin, class margin, strong-model disagreement, multiface conflict, and input issues\n"
        "- aliases in panels come from `strong_experiments.csv`\n"
        f"- theta_jesse={theta_jesse:.6f}\n"
        f"- theta_mila={theta_mila:.6f}\n",
        encoding="utf-8",
    )
    print(f"done: exported {len(merged)} champion-risk visualizations -> {out_root}")


if __name__ == "__main__":
    main()
