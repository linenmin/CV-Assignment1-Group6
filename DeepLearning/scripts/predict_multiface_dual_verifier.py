from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.inference.insightface_dual_verifier import (
    _load_lookalike_clusters,
    aggregate_multi_face_scores,
    loo_top_k_mean_for_identity,
    top_k_mean_cosine_similarity,
    youden_threshold,
)
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe


def _load_raw_bgr(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path), allow_pickle=False)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _read_bgr(image_path: Path) -> np.ndarray | None:
    if not image_path.is_file():
        return None
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


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


def _extract_all_face_embeddings(app, bgr: np.ndarray) -> list[np.ndarray]:
    faces = app.get(bgr)
    return [np.asarray(face.normed_embedding, dtype=np.float32).ravel() for face in faces]


def _collect_single_embeddings(config: dict[str, Any]):
    app, rec_model = _prepare_insightface_app(config)
    inf = config["inference"]
    embedding_mode = str(inf.get("insightface_embedding", "detect_align_with_crop_fallback")).strip().lower()
    splits_dir = project_path(config["data"]["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    fallback_count = 0
    id_to_emb: dict[int, np.ndarray] = {}

    for frame in (train_df, val_df, test_df):
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
                        faces = _extract_all_face_embeddings(app, bgr)
                        if faces:
                            emb = faces[0]
                if emb is None:
                    fallback_count += 1
                    crop_bgr = _read_bgr(project_path(str(row["image_path"])))
                    if crop_bgr is not None:
                        emb = _embed_direct_arcface(rec_model, crop_bgr)
            else:
                crop_bgr = _read_bgr(project_path(str(row["image_path"])))
                if crop_bgr is not None:
                    emb = _embed_direct_arcface(rec_model, crop_bgr)

            if emb is not None:
                id_to_emb[sid] = emb

    return train_df, val_df, test_df, id_to_emb, app, rec_model, fallback_count


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
    theta_jesse, meta_j = youden_threshold(scores_j_pos, scores_j_neg)

    scores_m_pos = np.array([loo_top_k_mean_for_identity(i, mila_all, top_k) for i in range(len(mila_all))], dtype=np.float64)
    scores_m_neg = np.array([top_k_mean_cosine_similarity(e, mila_gallery, top_k) for e in sarah_embs], dtype=np.float64)
    theta_mila, meta_m = youden_threshold(scores_m_pos, scores_m_neg)

    return jesse_gallery, mila_gallery, float(theta_jesse), float(theta_mila), meta_j, meta_m


def _probe_face_embeddings(row: pd.Series, app, rec_model) -> tuple[list[np.ndarray], str]:
    source_path = row.get("source_path", "")
    if source_path and str(source_path).endswith(".npy"):
        sp = project_path(str(source_path))
        if sp.is_file():
            bgr = _load_raw_bgr(sp)
            faces = _extract_all_face_embeddings(app, bgr)
            if faces:
                return faces, "detect_align"
    crop_bgr = _read_bgr(project_path(str(row["image_path"])))
    if crop_bgr is None:
        return [], "missing"
    return [_embed_direct_arcface(rec_model, crop_bgr)], "crop_fallback"


def _score_probe_frame_multiface(
    frame: pd.DataFrame,
    app,
    rec_model,
    jesse_gallery: np.ndarray,
    mila_gallery: np.ndarray,
    top_k: int,
    theta_jesse: float,
    theta_mila: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        face_embeddings, source = _probe_face_embeddings(row, app, rec_model)
        score_jesse = np.array([top_k_mean_cosine_similarity(emb, jesse_gallery, top_k) for emb in face_embeddings], dtype=np.float32)
        score_mila = np.array([top_k_mean_cosine_similarity(emb, mila_gallery, top_k) for emb in face_embeddings], dtype=np.float32)
        agg = aggregate_multi_face_scores(score_jesse, score_mila, theta_jesse, theta_mila)
        record = {
            "id": int(row["id"]),
            "pred_multiface": int(agg["prediction"]),
            "best_score_jesse": float(agg["best_score_jesse"]),
            "best_score_mila": float(agg["best_score_mila"]),
            "best_excess_jesse": float(agg["best_excess_jesse"]),
            "best_excess_mila": float(agg["best_excess_mila"]),
            "best_face_index_jesse": int(agg["best_face_index_jesse"]),
            "best_face_index_mila": int(agg["best_face_index_mila"]),
            "num_faces_detected": int(len(face_embeddings)),
            "probe_source": source,
        }
        if "class" in row:
            record["label"] = int(row["class"])
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-face detect_align dual verifier prototype")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike.yaml",
    )
    parser.add_argument(
        "--reference-submission",
        default="data/submissions/20260403_134351_exp_067_buffalo_l_detect_align_dual_verifier_lookalike_submission.csv",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    exp_name = config["experiment_name"]
    output_root = ensure_dir(project_path("outputs", exp_name))

    train_df, val_df, test_df, id_to_emb, app, rec_model, fallback_count = _collect_single_embeddings(config)
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
    val_scores = _score_probe_frame_multiface(
        frame=val_df,
        app=app,
        rec_model=rec_model,
        jesse_gallery=val_jg,
        mila_gallery=val_mg,
        top_k=top_k,
        theta_jesse=theta_j_val,
        theta_mila=theta_m_val,
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
    test_scores = _score_probe_frame_multiface(
        frame=test_df,
        app=app,
        rec_model=rec_model,
        jesse_gallery=test_jg,
        mila_gallery=test_mg,
        top_k=top_k,
        theta_jesse=theta_j_test,
        theta_mila=theta_m_test,
    )

    val_accuracy = float((val_scores["pred_multiface"].to_numpy(dtype=np.int64) == val_scores["label"].to_numpy(dtype=np.int64)).mean())
    val_face_hist = val_scores["num_faces_detected"].value_counts().sort_index().to_dict()
    test_face_hist = test_scores["num_faces_detected"].value_counts().sort_index().to_dict()

    val_scores.to_csv(output_root / "val_scores.csv", index=False)
    test_scores.to_csv(output_root / "test_scores.csv", index=False)

    submission = build_submission_dataframe(test_df, test_scores["pred_multiface"].astype(int).tolist())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_path("data", "submissions", f"{timestamp}_{exp_name}_submission.csv")
    save_submission_dataframe(submission, submission_path)

    reference = pd.read_csv(project_path(args.reference_submission)).set_index("id")["class"].astype(int)
    candidate = test_scores.set_index("id")["pred_multiface"].astype(int)
    aligned = pd.concat([reference.rename("ref"), candidate.rename("cand")], axis=1).dropna()
    diff = aligned[aligned["ref"] != aligned["cand"]]
    transitions = diff.assign(change=diff["ref"].astype(str) + "->" + diff["cand"].astype(str))["change"].value_counts().to_dict()

    metrics = {
        "mode": "insightface_dual_verifier_multiface",
        "reference_experiment": "exp_067_buffalo_l_detect_align_dual_verifier_lookalike",
        "gallery_top_k": top_k,
        "detect_align_fallback_count_gallery": fallback_count,
        "val_theta_jesse": theta_j_val,
        "val_theta_mila": theta_m_val,
        "test_theta_jesse": theta_j_test,
        "test_theta_mila": theta_m_test,
        "val_accuracy_multiface": val_accuracy,
        "val_calib_meta_jesse": meta_j_val,
        "val_calib_meta_mila": meta_m_val,
        "test_calib_meta_jesse": meta_j_test,
        "test_calib_meta_mila": meta_m_test,
        "val_num_diff_vs_067_style": int((val_scores["pred_multiface"].to_numpy(dtype=np.int64) != pd.read_csv(output_root.parent / "exp_067_buffalo_l_detect_align_dual_verifier_lookalike" / "prototype_metrics.json" if False else np.array([]))).shape[0]) if False else None,
        "val_face_count_histogram": {str(k): int(v) for k, v in val_face_hist.items()},
        "test_face_count_histogram": {str(k): int(v) for k, v in test_face_hist.items()},
        "num_test_diff_vs_reference_067": int(len(diff)),
        "test_diff_vs_reference_067": {str(k): int(v) for k, v in transitions.items()},
        "submission_path": str(submission_path),
    }
    (output_root / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": exp_name,
            "stage": "predict",
            "notes": "scripts/predict_multiface_dual_verifier.py::multiface_probe_maxpool",
            "submission_path": str(submission_path),
            "checkpoint_path": "insightface_buffalo_l_multiface_probe",
            "config_path": args.config,
        },
    )

    print(json.dumps(metrics, indent=2))
    print(f"submission 已生成: {submission_path}")


if __name__ == "__main__":
    main()
