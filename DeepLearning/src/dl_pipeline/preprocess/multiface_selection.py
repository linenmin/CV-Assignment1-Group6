from __future__ import annotations

from typing import Any

import numpy as np

from dl_pipeline.inference.insightface_dual_verifier import describe_multi_face_scores


def build_face_selection_rows(
    score_jesse: np.ndarray,
    score_mila: np.ndarray,
    theta_jesse: float,
    theta_mila: float,
) -> list[dict[str, Any]]:
    return describe_multi_face_scores(
        score_jesse=score_jesse,
        score_mila=score_mila,
        theta_jesse=theta_jesse,
        theta_mila=theta_mila,
    )


def select_face_index_for_label(rows: list[dict[str, Any]], label: int | None) -> int | None:
    if not rows:
        return None

    if label == 1:
        best = max(rows, key=lambda row: (float(row["score_jesse"]), float(row["excess_jesse"])))
        return int(best["face_index"])
    if label == 2:
        best = max(rows, key=lambda row: (float(row["score_mila"]), float(row["excess_mila"])))
        return int(best["face_index"])

    # other / unlabeled probe: pick the face with strongest target-like evidence.
    best = max(
        rows,
        key=lambda row: (
            max(float(row["excess_jesse"]), float(row["excess_mila"])),
            max(float(row["score_jesse"]), float(row["score_mila"])),
        ),
    )
    return int(best["face_index"])
