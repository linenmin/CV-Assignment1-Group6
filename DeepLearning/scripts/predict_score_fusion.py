from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.inference.prototype import (
    compute_class_prototypes,
    compute_similarity_matrix,
    fuse_similarity_matrices,
    predict_open_set_from_similarity_matrix,
)
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_path(path_value)


def _export_member_embeddings(member: dict[str, object], output_path: Path) -> Path:
    export_script = Path(__file__).resolve().parent / "export_embeddings.py"
    config_path = _resolve_path(str(member["config_path"]))
    checkpoint_path = _resolve_path(str(member["checkpoint_path"]))
    command = [
        sys.executable,
        str(export_script),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def _select_best_threshold_from_scores(
    similarity_matrix: torch.Tensor,
    labels: torch.Tensor,
    prototype_labels: list[int],
    other_label: int,
    threshold_values: list[float],
) -> tuple[float, float]:
    if not threshold_values:
        raise ValueError("threshold_values 不能为空。")

    best_threshold = threshold_values[0]
    best_accuracy = -1.0
    for threshold in threshold_values:
        predictions, _ = predict_open_set_from_similarity_matrix(
            similarity_matrix=similarity_matrix,
            prototype_labels=prototype_labels,
            other_label=other_label,
            threshold=threshold,
        )
        accuracy = (predictions == labels).float().mean().item()
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy
    return best_threshold, best_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="对不同 backbone 的 prototype score 做融合并生成 submission。")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    experiment_name = config["experiment_name"]
    fusion_config = config["fusion"]
    members = fusion_config["members"]
    prototype_labels = fusion_config.get("prototype_labels", [1, 2])
    other_label = fusion_config.get("other_label", 0)
    threshold_values = fusion_config.get("threshold_values", [0.55])
    fusion_method = fusion_config["method"]
    fusion_weights = fusion_config.get("weights")

    output_root = ensure_dir(project_path("outputs", experiment_name))
    cache_dir = ensure_dir(output_root / "fusion_cache")
    test_df = pd.read_csv(project_path(config["data"]["splits_dir"]) / "test.csv")

    val_similarity_matrices: list[torch.Tensor] = []
    test_similarity_matrices: list[torch.Tensor] = []
    member_cache_paths: list[str] = []
    reference_val_labels = None
    reference_test_ids = None

    for member in members:
        member_name = str(member["name"])
        cache_path = cache_dir / f"{member_name}_embeddings.pt"
        _export_member_embeddings(member, cache_path)
        member_cache_paths.append(str(cache_path))
        payload = torch.load(cache_path, map_location="cpu")

        train_embeddings = payload["train_embeddings"]
        train_labels = payload["train_labels"]
        val_embeddings = payload["val_embeddings"]
        val_labels = payload["val_labels"]
        test_embeddings = payload["test_embeddings"]
        test_ids = payload["test_ids"]

        if reference_val_labels is None:
            reference_val_labels = val_labels
            reference_test_ids = test_ids
        else:
            if not torch.equal(reference_val_labels, val_labels):
                raise ValueError("不同成员的 val_labels 不一致，不能做融合。")
            if not torch.equal(reference_test_ids, test_ids):
                raise ValueError("不同成员的 test_ids 不一致，不能做融合。")

        prototypes = compute_class_prototypes(
            train_embeddings,
            train_labels,
            prototype_labels=prototype_labels,
        )
        val_similarity, ordered_labels = compute_similarity_matrix(
            val_embeddings,
            prototypes,
            prototype_labels=prototype_labels,
        )
        test_similarity, ordered_test_labels = compute_similarity_matrix(
            test_embeddings,
            prototypes,
            prototype_labels=prototype_labels,
        )
        if ordered_labels != prototype_labels or ordered_test_labels != prototype_labels:
            raise ValueError("prototype label 顺序与 fusion 配置不一致。")

        val_similarity_matrices.append(val_similarity)
        test_similarity_matrices.append(test_similarity)

    fused_val_similarity = fuse_similarity_matrices(
        val_similarity_matrices,
        method=fusion_method,
        weights=fusion_weights,
    )
    fused_test_similarity = fuse_similarity_matrices(
        test_similarity_matrices,
        method=fusion_method,
        weights=fusion_weights,
    )

    selected_threshold, val_accuracy = _select_best_threshold_from_scores(
        similarity_matrix=fused_val_similarity,
        labels=reference_val_labels,
        prototype_labels=prototype_labels,
        other_label=other_label,
        threshold_values=threshold_values,
    )
    test_predictions, test_scores = predict_open_set_from_similarity_matrix(
        similarity_matrix=fused_test_similarity,
        prototype_labels=prototype_labels,
        other_label=other_label,
        threshold=selected_threshold,
    )

    metrics = {
        "mode": "score_fusion",
        "fusion_method": fusion_method,
        "fusion_weights": fusion_weights,
        "prototype_labels": prototype_labels,
        "other_label": other_label,
        "selected_threshold": selected_threshold,
        "val_accuracy": val_accuracy,
        "member_cache_paths": member_cache_paths,
        "members": members,
        "mean_test_best_score": float(test_scores.mean().item()),
    }
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    submission = build_submission_dataframe(test_df, test_predictions.tolist())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_path(
        config["data"]["submissions_dir"],
        f"{timestamp}_{experiment_name}_submission.csv",
    )
    save_submission_dataframe(submission, submission_path)

    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": experiment_name,
            "stage": "predict",
            "submission_path": str(submission_path),
            "checkpoint_path": json.dumps(member_cache_paths, ensure_ascii=False),
            "config_path": config["config_path"],
        },
    )
    print(f"fusion submission 已生成: {submission_path}")


if __name__ == "__main__":
    main()
