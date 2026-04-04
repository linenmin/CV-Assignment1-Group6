from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def rank_class_neighbors(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    gallery_ids: np.ndarray,
    target_label: int,
    top_k: int,
) -> list[dict[str, Any]]:
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    gallery_embeddings = np.asarray(gallery_embeddings, dtype=np.float32)
    gallery_labels = np.asarray(gallery_labels)
    gallery_ids = np.asarray(gallery_ids)

    if gallery_embeddings.ndim != 2:
        raise ValueError("gallery_embeddings 必须是二维矩阵。")
    if not (len(gallery_embeddings) == len(gallery_labels) == len(gallery_ids)):
        raise ValueError("gallery embeddings / labels / ids 长度必须一致。")

    mask = gallery_labels.astype(int) == int(target_label)
    if not np.any(mask):
        return []

    candidate_embeddings = gallery_embeddings[mask]
    candidate_ids = gallery_ids[mask]
    sims = candidate_embeddings @ query
    order = np.argsort(-sims)[: max(1, int(top_k))]
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "rank": int(rank),
                "id": int(candidate_ids[idx]),
                "label": int(target_label),
                "similarity": float(sims[idx]),
            }
        )
    return rows


def prepare_test_audit_index(audit_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"split", "id"}
    missing = required_columns.difference(audit_df.columns)
    if missing:
        raise ValueError(f"selection audit 缺少必要列: {sorted(missing)}")

    test_df = audit_df.loc[audit_df["split"].astype(str) == "test"].copy()
    if test_df.empty:
        raise ValueError("selection audit 中不存在 split=test 的记录。")

    if test_df["id"].duplicated().any():
        duplicated_ids = test_df.loc[test_df["id"].duplicated(), "id"].astype(int).tolist()
        raise ValueError(f"selection audit 的 test 子集存在重复 id: {duplicated_ids[:10]}")

    return test_df.set_index("id")
