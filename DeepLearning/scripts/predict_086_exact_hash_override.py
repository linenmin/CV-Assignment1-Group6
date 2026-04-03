from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.inference.exact_match_override import (
    apply_exact_hash_overrides,
    build_consistent_hash_label_map,
    compute_array_sha1,
)
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe


def _load_hash(path: Path) -> str:
    arr = np.load(path, allow_pickle=False)
    return compute_array_sha1(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply exact raw-image hash overrides on top of exp_086 submission.")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike.yaml",
    )
    parser.add_argument(
        "--base-submission",
        default="data/submissions/20260403_195050_exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike_submission.csv",
    )
    parser.add_argument(
        "--base-scores",
        default="outputs/exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike/test_scores.csv",
    )
    parser.add_argument(
        "--exp-name",
        default="exp_087_multiface_exact_raw_hash_override",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    splits_dir = project_path(config["data"]["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    base_submission = pd.read_csv(project_path(args.base_submission))
    base_scores = pd.read_csv(project_path(args.base_scores))

    labeled_pairs: list[tuple[str, int]] = []
    labeled_sources: dict[str, tuple[str, int]] = {}
    for split_name, frame in (("train", train_df), ("val", val_df)):
        for _, row in frame.iterrows():
            image_hash = _load_hash(project_path(str(row["source_path"])))
            labeled_pairs.append((image_hash, int(row["class"])))
            labeled_sources.setdefault(image_hash, (split_name, int(row["id"])))
    hash_to_label = build_consistent_hash_label_map(labeled_pairs)

    test_hashes: dict[int, str] = {}
    for _, row in test_df.iterrows():
        sid = int(row["id"])
        test_hashes[sid] = _load_hash(project_path(str(row["source_path"])))

    base_predictions = {int(row["id"]): int(row["class"]) for _, row in base_submission.iterrows()}
    updated_predictions, overridden = apply_exact_hash_overrides(base_predictions, test_hashes, hash_to_label)

    output_root = ensure_dir(project_path("outputs", args.exp_name))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_path("data", "submissions", f"{timestamp}_{args.exp_name}_submission.csv")
    ordered_predictions = [updated_predictions[int(sid)] for sid in test_df["id"].tolist()]
    updated_submission = build_submission_dataframe(test_df, ordered_predictions)
    save_submission_dataframe(updated_submission, submission_path)

    override_rows: list[dict[str, Any]] = []
    for sid, change in sorted(overridden.items()):
        source_split, source_id = labeled_sources[change["hash"]]
        score_row = base_scores.loc[base_scores["id"] == sid].iloc[0]
        override_rows.append(
            {
                "id": sid,
                "from": int(change["from"]),
                "to": int(change["to"]),
                "hash": str(change["hash"]),
                "matched_split": source_split,
                "matched_id": int(source_id),
                "probe_source": str(score_row["probe_source"]),
                "best_score_jesse": float(score_row["best_score_jesse"]),
                "best_score_mila": float(score_row["best_score_mila"]),
            }
        )
    override_df = pd.DataFrame(override_rows)
    override_df.to_csv(output_root / "overrides.csv", index=False)

    matched_count = sum(1 for sid, image_hash in test_hashes.items() if image_hash in hash_to_label)
    metrics = {
        "base_experiment": "exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike",
        "mode": "exact_raw_hash_override",
        "num_test_hash_matches_to_labeled": int(matched_count),
        "num_overrides_applied": int(len(overridden)),
        "override_transitions": override_df.assign(change=override_df["from"].astype(str) + "->" + override_df["to"].astype(str))["change"].value_counts().to_dict() if len(override_df) else {},
        "submission_path": str(submission_path),
    }
    (output_root / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": args.exp_name,
            "stage": "predict",
            "notes": "scripts/predict_086_exact_hash_override.py::inherit_label_for_exact_raw_duplicates",
            "submission_path": str(submission_path),
            "checkpoint_path": "exp_086_exact_hash_override",
            "config_path": args.config,
        },
    )

    print(json.dumps(metrics, indent=2))
    if len(override_df):
        print(override_df.to_string(index=False))
    print(f"submission 已生成: {submission_path}")


if __name__ == "__main__":
    main()
