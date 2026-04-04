from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.inference.insightface_dual_verifier import (
    _load_lookalike_clusters,
    aggregate_multi_face_scores,
    describe_multi_face_scores,
    loo_top_k_mean_for_identity,
    top_k_mean_cosine_similarity,
    youden_threshold,
)


PANEL_W = 1480
PANEL_H = 1040
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
KEY_EXPERIMENTS = [
    "exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta",
    "exp_067_buffalo_l_detect_align_dual_verifier_lookalike",
    "exp_082_vit061_buffalol067_gated_score_fusion_valgrid",
    "exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike",
    "exp_087_multiface_exact_raw_hash_override",
    "exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta",
    "exp_089_vit_adaface_multiface_selected_other_top2_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta",
    "exp_090_vit_adaface_multiface_selected_20ep_last4block_ce_neighborhoodaware_fixed055_hfliptta",
    "exp_091_vit_adaface_multiface_selected_50ep_last2block_ce_neighborhoodaware_fixed055_valloss_hfliptta",
    "exp_092_vit_adaface_multiface_selected_50ep_last4block_ce_neighborhoodaware_fixed055_valloss_hfliptta",
]


def _load_raw_bgr(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path), allow_pickle=False)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


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


def _normalize_entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    entropy = float(-(probs * np.log(probs)).sum())
    max_entropy = math.log(3.0)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _score_suspicion(num_unique: int, entropy_norm: float, top_share: float, champion_share: float) -> float:
    return (
        40.0 * float(num_unique - 1)
        + 40.0 * entropy_norm
        + 15.0 * (1.0 - top_share)
        + 5.0 * (1.0 - champion_share)
    )


def _submission_timestamp_and_name(path: Path) -> tuple[str, str]:
    match = re.match(r"^(\d{8}_\d{6})_(.+)_submission$", path.stem)
    if match:
        return match.group(1), match.group(2)
    return ("", path.stem)


def _load_latest_submissions(submissions_dir: Path) -> dict[str, Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for path in submissions_dir.glob("*_submission.csv"):
        timestamp, exp_name = _submission_timestamp_and_name(path)
        if exp_name not in latest or timestamp > latest[exp_name][0]:
            latest[exp_name] = (timestamp, path)
    return {exp_name: item[1] for exp_name, item in latest.items()}


def _build_prediction_matrix(submission_paths: dict[str, Path]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for exp_name in sorted(submission_paths):
        frame = pd.read_csv(submission_paths[exp_name])
        frame = frame[["id", "class"]].rename(columns={"class": exp_name})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="id", how="inner")
    if merged is None:
        raise FileNotFoundError("no submission files found")
    return merged.sort_values("id").reset_index(drop=True)


def _attach_disagreement_stats(pred_df: pd.DataFrame, champion_exp: str) -> pd.DataFrame:
    exp_cols = [col for col in pred_df.columns if col != "id"]
    rows: list[dict[str, Any]] = []
    for _, row in pred_df.iterrows():
        preds = row[exp_cols].astype(int).to_numpy()
        counts = np.bincount(preds, minlength=3)
        num_unique = int(np.count_nonzero(counts))
        entropy_norm = _normalize_entropy(counts)
        top_share = float(counts.max() / len(preds))
        sorted_counts = np.sort(counts)[::-1]
        runner_up_share = float(sorted_counts[1] / len(preds)) if len(sorted_counts) > 1 else 0.0
        champion_pred = int(row[champion_exp])
        champion_share = float(counts[champion_pred] / len(preds))
        majority_pred = int(np.argmax(counts))
        suspicion = _score_suspicion(num_unique, entropy_norm, top_share, champion_share)
        rows.append(
            {
                "id": int(row["id"]),
                "num_unique_preds": num_unique,
                "count_0": int(counts[0]),
                "count_1": int(counts[1]),
                "count_2": int(counts[2]),
                "entropy_norm": entropy_norm,
                "top_vote_share": top_share,
                "runner_up_share": runner_up_share,
                "majority_pred": majority_pred,
                "champion_pred": champion_pred,
                "champion_support_share": champion_share,
                "is_champion_minority": int(champion_share < top_share),
                "suspicion_score": suspicion,
            }
        )
    stats_df = pd.DataFrame(rows)
    return pred_df.merge(stats_df, on="id", how="left")


def _render_sample_canvas(
    raw_rgb: np.ndarray | None,
    haar_rgb: np.ndarray | None,
    mtcnn_rgb: np.ndarray | None,
    selected_rgb: np.ndarray | None,
    face_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    key_predictions: list[tuple[str, int | None]],
) -> np.ndarray:
    canvas = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    header = (
        f"rank={stats['rank']} | id={stats['id']} | suspicion={stats['suspicion_score']:.2f} "
        f"| unique={stats['num_unique_preds']} | votes 0/1/2={stats['count_0']}/{stats['count_1']}/{stats['count_2']}"
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
    crop_panels = [
        ("HAAR", haar_rgb),
        ("MTCNN", mtcnn_rgb),
        ("SEL", selected_rgb),
    ]
    for idx, (title, image_rgb) in enumerate(crop_panels):
        cv2.putText(canvas, title, (right_x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT, 2, cv2.LINE_AA)
        panel = _make_missing_panel(f"{title}\nmissing", right_w // 3 - 8, small_h) if image_rgb is None else _fit_rgb(image_rgb, right_w // 3 - 8, small_h)
        offset = idx * (right_w // 3)
        x0 = right_x + offset
        canvas[y + 28:y + 28 + small_h, x0:x0 + panel.shape[1]] = panel
    y += small_h + 54

    lines = [
        f"majority={stats['majority_pred']}  champion={stats['champion_pred']}  champion_support={stats['champion_support_share']:.3f}",
        f"entropy={stats['entropy_norm']:.3f}  top_share={stats['top_vote_share']:.3f}  runner_up={stats['runner_up_share']:.3f}",
        f"probe={stats['probe_source']}  faces={stats['num_faces_detected']}  multiface_pred={stats['multiface_pred']}",
        f"bestJ={stats['best_score_jesse']:.3f} ({stats['best_excess_jesse']:+.3f})  bestM={stats['best_score_mila']:.3f} ({stats['best_excess_mila']:+.3f})",
    ]
    cv2.putText(canvas, "Summary", (right_x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT, 2, cv2.LINE_AA)
    text_y = y + 46
    for line in lines:
        cv2.putText(canvas, line, (right_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT, 1, cv2.LINE_AA)
        text_y += 24

    cv2.putText(canvas, "Key Predictions", (right_x, text_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT, 2, cv2.LINE_AA)
    text_y += 36
    for exp_name, pred in key_predictions:
        short_name = exp_name.replace("exp_", "e")
        line = f"{short_name}: {pred if pred is not None else 'NA'}"
        cv2.putText(canvas, line, (right_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
        text_y += 22
        if text_y > PANEL_H - GAP - FACE_THUMB - 10:
            break

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
    parser = argparse.ArgumentParser(description="Export high-disagreement test sample visualizations from many submissions.")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike.yaml",
    )
    parser.add_argument(
        "--base-test-csv",
        default="data/splits/exp_001/test.csv",
    )
    parser.add_argument(
        "--selected-test-csv",
        default="data/splits/exp_088_multiface_selected/test.csv",
    )
    parser.add_argument(
        "--champion-exp",
        default="exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta",
    )
    parser.add_argument(
        "--submissions-dir",
        default="data/submissions",
    )
    parser.add_argument(
        "--out-dir",
        default="data/visualizations/test_disagreement_probe/top50_exp_088",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--ids", nargs="*", type=int, default=[])
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    out_root = ensure_dir(project_path(args.out_dir))
    images_root = ensure_dir(out_root / "images")

    submission_paths = _load_latest_submissions(project_path(args.submissions_dir))
    if args.champion_exp not in submission_paths:
        raise KeyError(f"champion submission not found: {args.champion_exp}")

    pred_df = _build_prediction_matrix(submission_paths)
    pred_df = _attach_disagreement_stats(pred_df, champion_exp=args.champion_exp)
    pred_df = pred_df.sort_values(
        ["num_unique_preds", "entropy_norm", "top_vote_share", "champion_support_share", "id"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    pred_df["rank"] = np.arange(1, len(pred_df) + 1)

    base_test_df = pd.read_csv(project_path(args.base_test_csv))
    selected_test_df = pd.read_csv(project_path(args.selected_test_csv)).rename(columns={"image_path": "selected_image_path"})
    meta_df = base_test_df.merge(selected_test_df[["id", "selected_image_path"]], on="id", how="left")
    ranked_df = pred_df.merge(meta_df, on="id", how="left")
    if args.ids:
        ranked_df = ranked_df[ranked_df["id"].isin(set(args.ids))].copy()
        ranked_df = ranked_df.sort_values(["rank", "id"]).reset_index(drop=True)
    else:
        ranked_df = ranked_df.head(int(args.top_n)).copy()

    app, rec_model = _prepare_insightface_app(config)
    train_df, val_df, id_to_emb = _collect_single_embeddings(config, app, rec_model)
    labeled_df = pd.concat([train_df, val_df], ignore_index=True)
    inf = config["inference"]
    top_k = int(inf.get("gallery_top_k", 5))
    jesse_label = int(inf.get("jesse_class", 1))
    mila_label = int(inf.get("mila_class", 2))
    other_label = int(inf.get("other_class", 0))
    michael_cluster = int(inf.get("michael_like_cluster", 1))
    sarah_cluster = int(inf.get("sarah_like_cluster", 0))
    look_df = _load_lookalike_clusters(project_path(inf["lookalike_assignments_csv"]))

    jesse_gallery, mila_gallery, theta_jesse, theta_mila = _calibrate_dual_verifier(
        gallery_df=labeled_df,
        calib_df=labeled_df,
        look_df=look_df,
        id_to_emb=id_to_emb,
        top_k=top_k,
        jesse_label=jesse_label,
        mila_label=mila_label,
        other_label=other_label,
        michael_cluster=michael_cluster,
        sarah_cluster=sarah_cluster,
    )

    mtcnn_dir = project_path("data", "processed", "exp_008_mtcnn_faces_112", "test")
    detail_rows: list[dict[str, Any]] = []

    for _, row in ranked_df.iterrows():
        sid = int(row["id"])
        raw_path = project_path(str(row["source_path"]))
        raw_rgb = _load_raw_rgb(raw_path) if raw_path.is_file() else None
        raw_bgr = _load_raw_bgr(raw_path) if raw_path.is_file() else None
        haar_rgb = _load_png_rgb(project_path(str(row["image_path"])))
        mtcnn_rgb = _load_png_rgb(mtcnn_dir / f"{sid}.png")
        selected_rgb = _load_png_rgb(project_path(str(row["selected_image_path"]))) if pd.notna(row["selected_image_path"]) else None

        face_rows = _extract_face_candidates(app, raw_bgr) if raw_bgr is not None else []
        probe_source = "detect_align" if face_rows else "missing"
        if not face_rows:
            fallback_bgr = _read_bgr(project_path(str(row["selected_image_path"]))) if pd.notna(row["selected_image_path"]) else None
            if fallback_bgr is None:
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

        score_jesse = np.array(
            [top_k_mean_cosine_similarity(face["embedding"], jesse_gallery, top_k) for face in face_rows],
            dtype=np.float32,
        )
        score_mila = np.array(
            [top_k_mean_cosine_similarity(face["embedding"], mila_gallery, top_k) for face in face_rows],
            dtype=np.float32,
        )
        agg = aggregate_multi_face_scores(score_jesse, score_mila, theta_jesse, theta_mila)
        score_rows = describe_multi_face_scores(score_jesse, score_mila, theta_jesse, theta_mila)

        key_predictions = []
        for exp_name in KEY_EXPERIMENTS:
            pred = int(row[exp_name]) if exp_name in row.index and pd.notna(row[exp_name]) else None
            key_predictions.append((exp_name, pred))

        stats = {
            "rank": int(row["rank"]),
            "id": sid,
            "suspicion_score": float(row["suspicion_score"]),
            "num_unique_preds": int(row["num_unique_preds"]),
            "count_0": int(row["count_0"]),
            "count_1": int(row["count_1"]),
            "count_2": int(row["count_2"]),
            "entropy_norm": float(row["entropy_norm"]),
            "top_vote_share": float(row["top_vote_share"]),
            "runner_up_share": float(row["runner_up_share"]),
            "majority_pred": int(row["majority_pred"]),
            "champion_pred": int(row["champion_pred"]),
            "champion_support_share": float(row["champion_support_share"]),
            "probe_source": probe_source,
            "num_faces_detected": int(len(face_rows)),
            "multiface_pred": int(agg["prediction"]),
            "best_score_jesse": float(agg["best_score_jesse"]),
            "best_score_mila": float(agg["best_score_mila"]),
            "best_excess_jesse": float(agg["best_excess_jesse"]),
            "best_excess_mila": float(agg["best_excess_mila"]),
        }

        for face_row, score_row in zip(face_rows, score_rows):
            detail_rows.append(
                {
                    "id": sid,
                    "rank": int(row["rank"]),
                    "probe_source": probe_source,
                    "bbox_x1": face_row["bbox"][0],
                    "bbox_y1": face_row["bbox"][1],
                    "bbox_x2": face_row["bbox"][2],
                    "bbox_y2": face_row["bbox"][3],
                    **score_row,
                }
            )

        canvas = _render_sample_canvas(
            raw_rgb=raw_rgb,
            haar_rgb=haar_rgb,
            mtcnn_rgb=mtcnn_rgb,
            selected_rgb=selected_rgb,
            face_rows=face_rows,
            score_rows=score_rows,
            stats=stats,
            key_predictions=key_predictions,
        )
        stem = (
            f"rank_{int(row['rank']):03d}_id{sid:04d}_u{int(row['num_unique_preds'])}"
            f"_maj{int(row['majority_pred'])}_champ{int(row['champion_pred'])}"
        )
        Image.fromarray(canvas).save(images_root / f"{stem}.png", format="PNG", compress_level=3)

    ranked_df.to_csv(out_root / "summary.csv", index=False)
    pred_df.to_csv(out_root / "all_ranked_summary.csv", index=False)
    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["rank", "id", "face_index"])
    detail_df.to_csv(out_root / "face_details.csv", index=False)
    pd.DataFrame(
        [{"slot": idx + 1, "experiment_name": exp_name} for idx, exp_name in enumerate(KEY_EXPERIMENTS)]
    ).to_csv(out_root / "key_experiments.csv", index=False)
    (out_root / "README.md").write_text(
        "# test disagreement probe\n\n"
        "- `images/`: top suspicious test samples ranked by cross-submission disagreement\n"
        "- `summary.csv`: selected sample ranking and all per-experiment predictions\n"
        "- `all_ranked_summary.csv`: full test-set ranking\n"
        "- `face_details.csv`: current multiface score details for selected ids\n"
        "- `key_experiments.csv`: prediction rows shown in the image panels\n"
        f"- submissions_used={len(submission_paths)}\n"
        f"- champion_exp={args.champion_exp}\n"
        f"- theta_jesse={theta_jesse:.6f}\n"
        f"- theta_mila={theta_mila:.6f}\n",
        encoding="utf-8",
    )
    print(f"done: exported {len(ranked_df)} suspicious test visualizations -> {out_root}")


if __name__ == "__main__":
    main()
