from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.inference.gated_fusion import (
    apply_agreement_aware_rescue_only_fusion,
    select_best_agreement_aware_rescue_only_params,
)
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe
from predict_061_067_gated_fusion import (
    _load_reference_predictions,
    _prediction_diff_summary,
    _resolve_checkpoint_path,
    _score_dual_verifier_tables,
    _score_neighborhood_tables,
)
from predict_061_067_rescueonly_labeledloo import _score_labeled_loo_061, _score_labeled_loo_067


def main() -> None:
    parser = argparse.ArgumentParser(description="exp_061 + exp_067 agreement-aware rescue-only fusion with labeled LOO tuning")
    parser.add_argument("--experiment-name", default="exp_085_vit061_buffalol067_agreementaware_rescueonly_labeledloo")
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml",
    )
    parser.add_argument("--primary-checkpoint", default="")
    parser.add_argument(
        "--secondary-config",
        default="configs/experiments/exp_067_buffalo_l_detect_align_dual_verifier_lookalike.yaml",
    )
    parser.add_argument(
        "--reference-primary-submission",
        default="data/submissions/20260402_224630_exp_061_vit_adaface_haar_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_submission.csv",
    )
    parser.add_argument(
        "--reference-secondary-submission",
        default="data/submissions/20260403_134351_exp_067_buffalo_l_detect_align_dual_verifier_lookalike_submission.csv",
    )
    parser.add_argument(
        "--reference-exp082-submission",
        default="data/submissions/20260403_175121_exp_082_vit061_buffalol067_gated_score_fusion_valgrid_submission.csv",
    )
    parser.add_argument(
        "--reference-exp083-submission",
        default="data/submissions/20260403_181210_exp_083_vit061_buffalol067_rescueonly_classaware_labeledloo_submission.csv",
    )
    parser.add_argument("--low-conf-threshold-values", nargs="+", type=float, default=[0.35, 0.40, 0.45, 0.50, 0.55])
    parser.add_argument("--rescue-margin-jesse-agree-values", nargs="+", type=float, default=[0.02, 0.06, 0.10, 0.14])
    parser.add_argument("--rescue-margin-jesse-disagree-values", nargs="+", type=float, default=[0.06, 0.10, 0.14, 0.18, 0.22])
    parser.add_argument("--rescue-margin-mila-agree-values", nargs="+", type=float, default=[0.02, 0.06, 0.10, 0.14])
    parser.add_argument("--rescue-margin-mila-disagree-values", nargs="+", type=float, default=[0.06, 0.10, 0.14, 0.18, 0.22])
    args = parser.parse_args()

    exp_name = args.experiment_name
    output_root = ensure_dir(project_path("outputs", exp_name))

    primary_config = load_experiment_config(args.primary_config)
    secondary_config = load_experiment_config(args.secondary_config)
    primary_checkpoint = _resolve_checkpoint_path(primary_config, args.primary_checkpoint or None)

    labeled_061 = _score_labeled_loo_061(primary_config, primary_checkpoint)
    labeled_067 = _score_labeled_loo_067(secondary_config)
    labeled_merged = labeled_061.merge(labeled_067, on=["id", "label"], how="inner", validate="one_to_one")

    best_params, search_records = select_best_agreement_aware_rescue_only_params(
        labels=labeled_merged["label"].to_numpy(dtype="int64"),
        anchor_predictions=labeled_merged["pred_061"].to_numpy(dtype="int64"),
        anchor_argmax=labeled_merged["pred_061_argmax"].to_numpy(dtype="int64"),
        anchor_scores=labeled_merged["score_061_final"].to_numpy(dtype="float32"),
        secondary_predictions=labeled_merged["pred_067"].to_numpy(dtype="int64"),
        secondary_excess_jesse=labeled_merged["score_067_excess_jesse"].to_numpy(dtype="float32"),
        secondary_excess_mila=labeled_merged["score_067_excess_mila"].to_numpy(dtype="float32"),
        other_label=0,
        low_conf_threshold_values=args.low_conf_threshold_values,
        rescue_margin_jesse_agree_values=args.rescue_margin_jesse_agree_values,
        rescue_margin_jesse_disagree_values=args.rescue_margin_jesse_disagree_values,
        rescue_margin_mila_agree_values=args.rescue_margin_mila_agree_values,
        rescue_margin_mila_disagree_values=args.rescue_margin_mila_disagree_values,
    )

    labeled_fused = apply_agreement_aware_rescue_only_fusion(
        anchor_predictions=labeled_merged["pred_061"].to_numpy(dtype="int64"),
        anchor_argmax=labeled_merged["pred_061_argmax"].to_numpy(dtype="int64"),
        anchor_scores=labeled_merged["score_061_final"].to_numpy(dtype="float32"),
        secondary_predictions=labeled_merged["pred_067"].to_numpy(dtype="int64"),
        secondary_excess_jesse=labeled_merged["score_067_excess_jesse"].to_numpy(dtype="float32"),
        secondary_excess_mila=labeled_merged["score_067_excess_mila"].to_numpy(dtype="float32"),
        other_label=0,
        low_conf_threshold=float(best_params["low_conf_threshold"]),
        rescue_margin_jesse_agree=float(best_params["rescue_margin_jesse_agree"]),
        rescue_margin_jesse_disagree=float(best_params["rescue_margin_jesse_disagree"]),
        rescue_margin_mila_agree=float(best_params["rescue_margin_mila_agree"]),
        rescue_margin_mila_disagree=float(best_params["rescue_margin_mila_disagree"]),
    )
    labeled_merged["pred_fused"] = labeled_fused

    _, test_061, metrics_061 = _score_neighborhood_tables(primary_config, primary_checkpoint)
    _, test_067, metrics_067 = _score_dual_verifier_tables(secondary_config)
    test_merged = test_061.merge(test_067, on="id", how="inner", validate="one_to_one")

    test_fused = apply_agreement_aware_rescue_only_fusion(
        anchor_predictions=test_merged["pred_061"].to_numpy(dtype="int64"),
        anchor_argmax=test_merged["pred_061_argmax"].to_numpy(dtype="int64"),
        anchor_scores=test_merged["score_061_final"].to_numpy(dtype="float32"),
        secondary_predictions=test_merged["pred_067"].to_numpy(dtype="int64"),
        secondary_excess_jesse=test_merged["score_067_excess_jesse"].to_numpy(dtype="float32"),
        secondary_excess_mila=test_merged["score_067_excess_mila"].to_numpy(dtype="float32"),
        other_label=0,
        low_conf_threshold=float(best_params["low_conf_threshold"]),
        rescue_margin_jesse_agree=float(best_params["rescue_margin_jesse_agree"]),
        rescue_margin_jesse_disagree=float(best_params["rescue_margin_jesse_disagree"]),
        rescue_margin_mila_agree=float(best_params["rescue_margin_mila_agree"]),
        rescue_margin_mila_disagree=float(best_params["rescue_margin_mila_disagree"]),
    )
    test_merged["pred_fused"] = test_fused

    (output_root / "labeled_loo_scores.csv").write_text(labeled_merged.to_csv(index=False), encoding="utf-8")
    (output_root / "test_scores.csv").write_text(test_merged.to_csv(index=False), encoding="utf-8")
    (output_root / "grid_search.json").write_text(json.dumps(search_records, indent=2), encoding="utf-8")

    test_df = pd.read_csv(project_path(primary_config["data"]["splits_dir"]) / "test.csv")
    submission = build_submission_dataframe(test_df, test_fused.tolist())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_path(primary_config["data"]["submissions_dir"], f"{timestamp}_{exp_name}_submission.csv")
    save_submission_dataframe(submission, submission_path)

    ref_061 = _load_reference_predictions(args.reference_primary_submission)
    ref_067 = _load_reference_predictions(args.reference_secondary_submission)
    ref_082 = _load_reference_predictions(args.reference_exp082_submission)
    ref_083 = _load_reference_predictions(args.reference_exp083_submission)
    pred_fused_series = test_merged.set_index("id")["pred_fused"].astype(int)

    diff_vs_061 = _prediction_diff_summary(ref_061, pred_fused_series)
    diff_vs_067 = _prediction_diff_summary(ref_067, pred_fused_series)
    diff_vs_082 = _prediction_diff_summary(ref_082, pred_fused_series)
    diff_vs_083 = _prediction_diff_summary(ref_083, pred_fused_series)

    labeled_rescue_mask = (labeled_merged["pred_061"] == 0) & (labeled_merged["pred_067"] != 0)
    labeled_agree_mask = labeled_rescue_mask & (labeled_merged["pred_061_argmax"] == labeled_merged["pred_067"])
    test_rescue_mask = (test_merged["pred_061"] == 0) & (test_merged["pred_067"] != 0)
    test_agree_mask = test_rescue_mask & (test_merged["pred_061_argmax"] == test_merged["pred_067"])

    metrics = {
        "mode": "agreement_aware_rescue_only_classaware_labeled_loo",
        "primary_experiment": primary_config["experiment_name"],
        "secondary_experiment": secondary_config["experiment_name"],
        "primary_checkpoint": str(primary_checkpoint),
        **metrics_061,
        **metrics_067,
        "best_params": best_params,
        "labeled_loo_accuracy_061": float((labeled_merged["pred_061"].to_numpy(dtype="int64") == labeled_merged["label"].to_numpy(dtype="int64")).mean()),
        "labeled_loo_accuracy_067": float((labeled_merged["pred_067"].to_numpy(dtype="int64") == labeled_merged["label"].to_numpy(dtype="int64")).mean()),
        "labeled_loo_accuracy_fused": float((labeled_fused == labeled_merged["label"].to_numpy(dtype="int64")).mean()),
        "labeled_loo_rescue_subset_size": int(labeled_rescue_mask.sum()),
        "labeled_loo_agree_subset_size": int(labeled_agree_mask.sum()),
        "labeled_loo_disagree_subset_size": int((labeled_rescue_mask & ~labeled_agree_mask).sum()),
        "labeled_loo_num_changed_vs_061": int((labeled_fused != labeled_merged["pred_061"].to_numpy(dtype="int64")).sum()),
        "test_rescue_subset_size": int(test_rescue_mask.sum()),
        "test_agree_subset_size": int(test_agree_mask.sum()),
        "test_disagree_subset_size": int((test_rescue_mask & ~test_agree_mask).sum()),
        "num_test_changed_vs_061": int((test_fused != test_merged["pred_061"].to_numpy(dtype="int64")).sum()),
        "num_test_changed_vs_082": int((test_fused != ref_082.reindex(test_merged["id"]).to_numpy(dtype="int64")).sum()),
        "num_test_changed_vs_083": int((test_fused != ref_083.reindex(test_merged["id"]).to_numpy(dtype="int64")).sum()),
        "fused_diff_vs_reference_061": diff_vs_061,
        "fused_diff_vs_reference_067": diff_vs_067,
        "fused_diff_vs_reference_082": diff_vs_082,
        "fused_diff_vs_reference_083": diff_vs_083,
        "search_space_size": len(search_records),
        "submission_path": str(submission_path),
    }
    (output_root / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": exp_name,
            "stage": "predict",
            "notes": "scripts/predict_061_067_agreement_rescue_labeledloo.py::labeled_loo_grid",
            "submission_path": str(submission_path),
            "checkpoint_path": f"{primary_checkpoint} | insightface_buffalo_l_agreement_rescue",
            "config_path": f"{args.primary_config} | {args.secondary_config}",
        },
    )

    print(json.dumps(metrics, indent=2))
    print(f"submission 已生成: {submission_path}")


if __name__ == "__main__":
    main()
