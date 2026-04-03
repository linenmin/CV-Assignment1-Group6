from __future__ import annotations

import argparse
from pathlib import Path
import sys
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


PANEL_W = 1280
PANEL_H = 920
RAW_BOX_W = 760
HEADER_H = 70
GAP = 16
FACE_THUMB = 112
TEXT = (25, 25, 25)
BG = (248, 248, 248)
BOX_DEFAULT = (120, 120, 120)
BOX_JESSE = (40, 120, 255)
BOX_MILA = (255, 120, 40)
BOX_BOTH = (30, 180, 70)


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
        raise ImportError("需要在 gpu_env 中安装 insightface 和 onnxruntime-gpu。") from exc

    inf = config["inference"]
    providers = inf.get("onnx_providers")
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name=inf.get("insightface_name", "buffalo_l"), providers=providers)
    app.prepare(ctx_id=int(inf.get("ctx_id", 0)), det_size=tuple(inf.get("det_size", [640, 640])))
    rec_model = app.models["recognition"]
    return app, rec_model


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
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        x = max(8, (width - size[0]) // 2)
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1, cv2.LINE_AA)
        y += 28
    return panel


def _draw_header(canvas: np.ndarray, text: str) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], HEADER_H), (235, 235, 235), -1)
    cv2.putText(canvas, text, (GAP, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, TEXT, 2, cv2.LINE_AA)


def _render_sample_canvas(
    raw_rgb: np.ndarray | None,
    haar_rgb: np.ndarray | None,
    mtcnn_rgb: np.ndarray | None,
    face_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    record: dict[str, Any],
) -> np.ndarray:
    canvas = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    header = (
        f"{record['split']} | id={record['id']} | label={record['label']} | pred={record['pred_multiface']} "
        f"| correct={record['correct']} | faces={record['num_faces_detected']} | source={record['probe_source']}"
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
    small_h = 180
    for title, image_rgb in (("HAAR", haar_rgb), ("MTCNN", mtcnn_rgb)):
        cv2.putText(canvas, title, (right_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)
        panel = _make_missing_panel(f"{title}\nmissing", right_w, small_h) if image_rgb is None else _fit_rgb(image_rgb, right_w, small_h)
        canvas[y + 30:y + 30 + small_h, right_x:right_x + right_w] = panel
        y += small_h + 48

    cv2.putText(canvas, "Face Scores", (right_x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)
    table_y = y + 50
    for score_row in score_rows[:8]:
        flags: list[str] = []
        if score_row["is_best_jesse"]:
            flags.append("bestJ")
        if score_row["is_best_mila"]:
            flags.append("bestM")
        if score_row["passes_jesse"]:
            flags.append("passJ")
        if score_row["passes_mila"]:
            flags.append("passM")
        line = (
            f"F{score_row['face_index']}  "
            f"J={score_row['score_jesse']:.3f} ({score_row['excess_jesse']:+.3f})  "
            f"M={score_row['score_mila']:.3f} ({score_row['excess_mila']:+.3f})  "
            f"{' '.join(flags)}"
        )
        cv2.putText(canvas, line, (right_x, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT, 1, cv2.LINE_AA)
        table_y += 26

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
            if x + FACE_THUMB > PANEL_W - GAP:
                break
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Export exp_086 multi-face probe visualizations for labeled samples.")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="data/visualizations/multiface_probe/exp_086_train_val_exp_001",
    )
    parser.add_argument("--ids", nargs="*", type=int, default=[])
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    out_root = ensure_dir(project_path(args.out_dir))
    images_root = ensure_dir(out_root / "images")

    app, rec_model = _prepare_insightface_app(config)
    train_df, val_df, id_to_emb = _collect_single_embeddings(config, app, rec_model)
    labeled_df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True)
    if args.ids:
        labeled_df = labeled_df[labeled_df["id"].astype(int).isin(set(args.ids))].copy()

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

    mtcnn_dir = project_path("data", "processed", "exp_008_mtcnn_faces_112")
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for export_idx, (_, row) in enumerate(labeled_df.iterrows()):
        sid = int(row["id"])
        raw_path = project_path(str(row["source_path"]))
        raw_rgb = _load_raw_rgb(raw_path) if raw_path.is_file() else None
        raw_bgr = _load_raw_bgr(raw_path) if raw_path.is_file() else None

        face_rows = _extract_face_candidates(app, raw_bgr) if raw_bgr is not None else []
        probe_source = "detect_align" if face_rows else "missing"
        if not face_rows:
            crop_bgr = _read_bgr(project_path(str(row["image_path"])))
            if crop_bgr is not None:
                h, w = crop_bgr.shape[:2]
                face_rows = [
                    {
                        "face_index": 0,
                        "bbox": [0.0, 0.0, float(w), float(h)],
                        "embedding": _embed_direct_arcface(rec_model, crop_bgr),
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

        haar_rgb = _load_png_rgb(project_path(str(row["image_path"])))
        split_dir = "train" if str(row["split"]) == "val" else str(row["split"])
        mtcnn_rgb = _load_png_rgb(mtcnn_dir / split_dir / f"{sid}.png")

        record = {
            "split": str(row["split"]),
            "id": sid,
            "label": int(row["class"]),
            "pred_multiface": int(agg["prediction"]),
            "correct": int(int(agg["prediction"]) == int(row["class"])),
            "num_faces_detected": int(len(face_rows)),
            "probe_source": probe_source,
            "best_face_index_jesse": int(agg["best_face_index_jesse"]),
            "best_face_index_mila": int(agg["best_face_index_mila"]),
            "best_score_jesse": float(agg["best_score_jesse"]),
            "best_score_mila": float(agg["best_score_mila"]),
            "best_excess_jesse": float(agg["best_excess_jesse"]),
            "best_excess_mila": float(agg["best_excess_mila"]),
        }
        summary_rows.append(record)

        for face_row, score_row in zip(face_rows, score_rows):
            detail_rows.append(
                {
                    "split": str(row["split"]),
                    "id": sid,
                    "label": int(row["class"]),
                    "pred_multiface": int(agg["prediction"]),
                    "probe_source": probe_source,
                    "bbox_x1": face_row["bbox"][0],
                    "bbox_y1": face_row["bbox"][1],
                    "bbox_x2": face_row["bbox"][2],
                    "bbox_y2": face_row["bbox"][3],
                    **score_row,
                }
            )

        canvas = _render_sample_canvas(raw_rgb, haar_rgb, mtcnn_rgb, face_rows, score_rows, record)
        stem = f"{record['split']}_{export_idx:03d}_id{sid:03d}_label{record['label']}_pred{record['pred_multiface']}"
        Image.fromarray(canvas).save(images_root / f"{stem}.png", format="PNG", compress_level=3)

    summary_df = pd.DataFrame(summary_rows).sort_values(["correct", "split", "id"], ascending=[True, True, True])
    detail_df = pd.DataFrame(detail_rows).sort_values(["split", "id", "face_index"])
    summary_df.to_csv(out_root / "summary.csv", index=False)
    detail_df.to_csv(out_root / "face_details.csv", index=False)
    (out_root / "README.md").write_text(
        "# exp_086 multi-face audit\n\n"
        "- `images/`: 每张 labeled 样本的多脸检测、分数和 face crop 可视化\n"
        "- `summary.csv`: 图片级预测摘要\n"
        "- `face_details.csv`: 每张脸的 bbox 与 Jesse/Mila 分数明细\n"
        f"- theta_jesse={theta_jesse:.6f}\n"
        f"- theta_mila={theta_mila:.6f}\n",
        encoding="utf-8",
    )
    print(f"完成: 导出 {len(summary_df)} 张 multiface 可视化 -> {out_root}")


if __name__ == "__main__":
    main()
