from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.inference.insightface_dual_verifier import (
    _load_lookalike_clusters,
    loo_top_k_mean_for_identity,
    top_k_mean_cosine_similarity,
    youden_threshold,
)
from dl_pipeline.preprocess.multiface_selection import build_face_selection_rows, select_face_index_for_label


@dataclass
class SelectedFaceArtifacts:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    audit_df: pd.DataFrame
    theta_jesse: float
    theta_mila: float


def _load_raw_bgr(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path), allow_pickle=False)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _read_bgr(image_path: Path) -> np.ndarray | None:
    if not image_path.is_file():
        return None
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return (v / n).astype(np.float32)


def _prepare_insightface_app(selection_cfg: dict[str, Any]):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError("需要在 gpu_env 中安装 insightface 和 onnxruntime-gpu。") from exc

    providers = selection_cfg.get("onnx_providers")
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name=selection_cfg.get("insightface_name", "buffalo_l"), providers=providers)
    app.prepare(
        ctx_id=int(selection_cfg.get("ctx_id", 0)),
        det_size=tuple(selection_cfg.get("det_size", [640, 640])),
    )
    rec_model = app.models["recognition"]
    try:
        from insightface.utils import face_align
    except ImportError as exc:
        raise ImportError("insightface.utils.face_align 不可用。") from exc
    return app, rec_model, face_align


def _embed_direct_arcface(rec_model, bgr: np.ndarray) -> np.ndarray:
    img112 = cv2.resize(bgr, (112, 112))
    raw = rec_model.get_feat(img112)
    return _l2_normalize(raw)


def _collect_single_embeddings(
    labeled_df: pd.DataFrame,
    app,
    rec_model,
    excluded_ids: set[int],
) -> dict[int, np.ndarray]:
    id_to_emb: dict[int, np.ndarray] = {}
    for _, row in labeled_df.iterrows():
        sid = int(row["id"])
        if sid in excluded_ids or sid in id_to_emb:
            continue
        emb = None
        source_path = row.get("source_path", "")
        if source_path and str(source_path).endswith(".npy"):
            sp = project_path(str(source_path))
            if sp.is_file():
                bgr = _load_raw_bgr(sp)
                faces = app.get(bgr)
                if faces:
                    emb = np.asarray(faces[0].normed_embedding, dtype=np.float32).ravel()
        if emb is None:
            crop_bgr = _read_bgr(project_path(str(row["image_path"])))
            if crop_bgr is not None:
                emb = _embed_direct_arcface(rec_model, crop_bgr)
        if emb is not None:
            id_to_emb[sid] = emb
    return id_to_emb


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
    labeled_df: pd.DataFrame,
    look_df: pd.DataFrame,
    id_to_emb: dict[int, np.ndarray],
    selection_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    top_k = int(selection_cfg.get("gallery_top_k", 5))
    jesse_label = int(selection_cfg.get("jesse_class", 1))
    mila_label = int(selection_cfg.get("mila_class", 2))
    other_label = int(selection_cfg.get("other_class", 0))
    michael_cluster = int(selection_cfg.get("michael_like_cluster", 1))
    sarah_cluster = int(selection_cfg.get("sarah_like_cluster", 0))

    jesse_all = _stack_identity(labeled_df, jesse_label, id_to_emb)
    mila_all = _stack_identity(labeled_df, mila_label, id_to_emb)
    emb_dim = int(next(iter(id_to_emb.values())).shape[0])
    jesse_gallery = np.stack(jesse_all, axis=0) if jesse_all else np.zeros((0, emb_dim), dtype=np.float32)
    mila_gallery = np.stack(mila_all, axis=0) if mila_all else np.zeros((0, emb_dim), dtype=np.float32)

    calib_other_ids = labeled_df[labeled_df["class"] == other_label]["id"].astype(int).tolist()
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


def _extract_face_candidates(raw_bgr: np.ndarray, app, face_align) -> list[dict[str, Any]]:
    faces = app.get(raw_bgr)
    candidates: list[dict[str, Any]] = []
    for idx, face in enumerate(faces):
        aligned = face_align.norm_crop(raw_bgr, landmark=face.kps, image_size=112)
        candidates.append(
            {
                "face_index": int(idx),
                "embedding": np.asarray(face.normed_embedding, dtype=np.float32).ravel(),
                "aligned_bgr": aligned,
            }
        )
    return candidates


def _score_candidates(
    candidates: list[dict[str, Any]],
    jesse_gallery: np.ndarray,
    mila_gallery: np.ndarray,
    theta_jesse: float,
    theta_mila: float,
    top_k: int,
) -> list[dict[str, Any]]:
    score_jesse = np.array(
        [top_k_mean_cosine_similarity(candidate["embedding"], jesse_gallery, top_k) for candidate in candidates],
        dtype=np.float32,
    )
    score_mila = np.array(
        [top_k_mean_cosine_similarity(candidate["embedding"], mila_gallery, top_k) for candidate in candidates],
        dtype=np.float32,
    )
    return build_face_selection_rows(score_jesse, score_mila, theta_jesse, theta_mila)


def _relative_project_path(path: Path) -> str:
    return str(path.relative_to(project_path()))


def build_multiface_selected_dataset(config: dict[str, Any]) -> SelectedFaceArtifacts:
    selection_cfg = config["multiface_selection"]
    base_splits_dir = project_path(selection_cfg.get("base_splits_dir", "data/splits/exp_001"))
    processed_dir = ensure_dir(project_path(config["data"]["processed_dir"]))
    splits_dir = ensure_dir(project_path(config["data"]["splits_dir"]))
    excluded_ids = {int(v) for v in selection_cfg.get("excluded_labeled_ids", [])}

    train_df = pd.read_csv(base_splits_dir / "train.csv")
    val_df = pd.read_csv(base_splits_dir / "val.csv")
    test_df = pd.read_csv(base_splits_dir / "test.csv")
    labeled_df = pd.concat([train_df, val_df], ignore_index=True)
    labeled_for_gallery = labeled_df[~labeled_df["id"].astype(int).isin(excluded_ids)].copy().reset_index(drop=True)

    app, rec_model, face_align = _prepare_insightface_app(selection_cfg)
    id_to_emb = _collect_single_embeddings(labeled_for_gallery, app, rec_model, excluded_ids)
    look_df = _load_lookalike_clusters(project_path(selection_cfg["lookalike_assignments_csv"]))
    jesse_gallery, mila_gallery, theta_jesse, theta_mila = _calibrate_dual_verifier(
        labeled_for_gallery, look_df, id_to_emb, selection_cfg
    )
    top_k = int(selection_cfg.get("gallery_top_k", 5))

    audit_rows: list[dict[str, Any]] = []

    def process_split(name: str, frame: pd.DataFrame, has_targets: bool) -> pd.DataFrame:
        out_records: list[dict[str, Any]] = []
        out_dir = ensure_dir(processed_dir / name)
        for _, row in frame.iterrows():
            sid = int(row["id"])
            if has_targets and sid in excluded_ids:
                continue

            raw_bgr = _load_raw_bgr(project_path(str(row["source_path"])))
            candidates = _extract_face_candidates(raw_bgr, app, face_align)
            score_rows = _score_candidates(candidates, jesse_gallery, mila_gallery, theta_jesse, theta_mila, top_k)
            label = int(row["class"]) if has_targets else None
            selected_index = select_face_index_for_label(score_rows, label)

            selection_source = "detect_align_selected"
            selected_bgr = None
            selected_row: dict[str, Any] | None = None
            if selected_index is not None and candidates:
                selected_bgr = candidates[selected_index]["aligned_bgr"]
                selected_row = next(item for item in score_rows if int(item["face_index"]) == int(selected_index))
            else:
                fallback_bgr = _read_bgr(project_path(str(row["image_path"])))
                if fallback_bgr is None:
                    raise FileNotFoundError(f"无法读取 fallback 裁剪图: {row['image_path']}")
                selected_bgr = cv2.resize(fallback_bgr, (112, 112), interpolation=cv2.INTER_AREA)
                selection_source = "crop_fallback"

            out_path = out_dir / f"{sid}.png"
            cv2.imwrite(str(out_path), selected_bgr)
            record = {
                "id": sid,
                "image_path": _relative_project_path(out_path),
                "source_path": str(row["source_path"]),
            }
            if has_targets:
                record["class"] = int(row["class"])
            out_records.append(record)

            audit_rows.append(
                {
                    "split": name,
                    "id": sid,
                    "label": int(row["class"]) if has_targets else None,
                    "selection_source": selection_source,
                    "num_faces_detected": int(len(candidates)),
                    "selected_face_index": int(selected_index) if selected_index is not None else -1,
                    "selected_score_jesse": float(selected_row["score_jesse"]) if selected_row else np.nan,
                    "selected_score_mila": float(selected_row["score_mila"]) if selected_row else np.nan,
                    "selected_excess_jesse": float(selected_row["excess_jesse"]) if selected_row else np.nan,
                    "selected_excess_mila": float(selected_row["excess_mila"]) if selected_row else np.nan,
                }
            )
        return pd.DataFrame(out_records)

    processed_train = process_split("train", train_df, has_targets=True)
    processed_val = process_split("val", val_df, has_targets=True)
    processed_test = process_split("test", test_df, has_targets=False)

    processed_train.to_csv(splits_dir / "train.csv", index=False)
    processed_val.to_csv(splits_dir / "val.csv", index=False)
    processed_test.to_csv(splits_dir / "test.csv", index=False)
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(processed_dir / "selection_audit.csv", index=False)

    return SelectedFaceArtifacts(
        train_df=processed_train,
        val_df=processed_val,
        test_df=processed_test,
        audit_df=audit_df,
        theta_jesse=theta_jesse,
        theta_mila=theta_mila,
    )
