from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.data.dataset import FaceDataset
from dl_pipeline.data.transforms import build_eval_transform
from dl_pipeline.inference.gated_fusion import (
    apply_rescue_only_fusion,
    select_best_rescue_only_params,
)
from dl_pipeline.inference.prototype import compute_class_prototypes, neighborhood_aware_predictions
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe
from predict_061_067_gated_fusion import (
    _calibrate_dual_verifier,
    _collect_embeddings,
    _collect_insightface_embeddings,
    _load_reference_predictions,
    _prediction_diff_summary,
    _resolve_checkpoint_path,
    _score_dual_verifier_frame,
    _score_dual_verifier_tables,
    _score_neighborhood_tables,
    _load_model_from_checkpoint,
)


def _collect_frame_embeddings_deterministic(
    model,
    frame: pd.DataFrame,
    image_size: int,
    normalization: str,
    batch_size: int,
    num_workers: int,
    device,
    use_horizontal_flip_tta: bool,
):
    dataset = FaceDataset(
        frame=frame,
        transform=build_eval_transform(image_size=image_size, normalization=normalization),
        has_targets=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    embeddings, labels = _collect_embeddings(model, loader, device, use_horizontal_flip_tta)
    ids = torch.as_tensor(frame["id"].astype(int).tolist(), dtype=torch.long)
    return embeddings, labels, ids


def _score_labeled_loo_061(
    primary_config: dict,
    primary_checkpoint: Path,
) -> pd.DataFrame:
    train_df = pd.read_csv(project_path(primary_config["data"]["splits_dir"]) / "train.csv")
    val_df = pd.read_csv(project_path(primary_config["data"]["splits_dir"]) / "val.csv")
    labeled_df = pd.concat([train_df, val_df], ignore_index=True).sort_values("id", kind="mergesort").reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() and primary_config["train"].get("accelerator", "gpu") != "cpu" else "cpu")
    model = _load_model_from_checkpoint(primary_config, primary_checkpoint).to(device)

    inf = primary_config["inference"]
    prototype_labels = inf.get("prototype_labels", [1, 2])
    other_label = int(inf.get("other_label", 0))
    threshold = float(inf.get("threshold", 0.55))
    nb = inf.get("neighborhood_aware", {})
    top_k = int(nb.get("top_k", 15))
    base_weight = float(nb.get("base_weight", 0.5))
    use_horizontal_flip_tta = bool(inf.get("tta_horizontal_flip", False))

    embeddings, labels, ids = _collect_frame_embeddings_deterministic(
        model=model,
        frame=labeled_df,
        image_size=int(primary_config["data"]["face_size"]),
        normalization=primary_config["data"]["normalization"],
        batch_size=int(primary_config["train"]["batch_size"]),
        num_workers=int(primary_config["data"]["num_workers"]),
        device=device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )

    records: list[dict[str, float | int]] = []
    n = embeddings.shape[0]
    for idx in range(n):
        keep = torch.ones(n, dtype=torch.bool)
        keep[idx] = False
        prototypes = compute_class_prototypes(
            embeddings[keep],
            labels[keep],
            prototype_labels=prototype_labels,
        )
        argmax_all, base_all, neighbor_all, final_all = neighborhood_aware_predictions(
            query_embeddings=embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=-1e9,
            top_k=top_k,
            base_weight=base_weight,
        )
        open_all, _, _, _ = neighborhood_aware_predictions(
            query_embeddings=embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=threshold,
            top_k=top_k,
            base_weight=base_weight,
        )
        records.append(
            {
                "id": int(ids[idx].item()),
                "label": int(labels[idx].item()),
                "pred_061_argmax": int(argmax_all[idx].item()),
                "pred_061": int(open_all[idx].item()),
                "score_061_base": float(base_all[idx].item()),
                "score_061_neighbor": float(neighbor_all[idx].item()),
                "score_061_final": float(final_all[idx].item()),
            }
        )

    return pd.DataFrame(records).sort_values("id", kind="mergesort").reset_index(drop=True)


def _score_labeled_loo_067(secondary_config: dict) -> pd.DataFrame:
    train_df, val_df, _, id_to_emb, _ = _collect_insightface_embeddings(secondary_config)
    labeled_df = pd.concat([train_df, val_df], ignore_index=True).sort_values("id", kind="mergesort").reset_index(drop=True)

    inf = secondary_config["inference"]
    top_k = int(inf.get("gallery_top_k", 5))
    jesse_label = int(inf.get("jesse_class", 1))
    mila_label = int(inf.get("mila_class", 2))
    other_label = int(inf.get("other_class", 0))
    michael_cluster = int(inf.get("michael_like_cluster", 1))
    sarah_cluster = int(inf.get("sarah_like_cluster", 0))
    look_df = pd.read_csv(project_path(inf["lookalike_assignments_csv"]))

    rows: list[pd.DataFrame] = []
    for idx in range(len(labeled_df)):
        sample = labeled_df.iloc[[idx]].copy()
        gallery_df = labeled_df.drop(index=idx).reset_index(drop=True)
        jesse_gallery, mila_gallery, theta_jesse, theta_mila, _, _ = _calibrate_dual_verifier(
            gallery_df=gallery_df,
            calib_df=gallery_df,
            look_df=look_df,
            id_to_emb=id_to_emb,
            top_k=top_k,
            jesse_label=jesse_label,
            mila_label=mila_label,
            other_label=other_label,
            michael_cluster=michael_cluster,
            sarah_cluster=sarah_cluster,
        )
        row_table = _score_dual_verifier_frame(
            frame=sample,
            id_to_emb=id_to_emb,
            jesse_gallery=jesse_gallery,
            mila_gallery=mila_gallery,
            top_k=top_k,
            theta_jesse=theta_jesse,
            theta_mila=theta_mila,
            other_label=other_label,
        )
        rows.append(row_table)

    return pd.concat(rows, ignore_index=True).sort_values("id", kind="mergesort").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="exp_061 + exp_067 rescue-only class-aware fusion with labeled LOO tuning")
    parser.add_argument("--experiment-name", default="exp_083_vit061_buffalol067_rescueonly_classaware_labeledloo")
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
    parser.add_argument("--low-conf-threshold-values", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    parser.add_argument("--rescue-margin-jesse-values", nargs="+", type=float, default=[0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30])
    parser.add_argument("--rescue-margin-mila-values", nargs="+", type=float, default=[0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30])
    parser.add_argument("--require-anchor-argmax-match-options", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()

    exp_name = args.experiment_name
    output_root = ensure_dir(project_path("outputs", exp_name))

    primary_config = load_experiment_config(args.primary_config)
    secondary_config = load_experiment_config(args.secondary_config)
    primary_checkpoint = _resolve_checkpoint_path(primary_config, args.primary_checkpoint or None)

    labeled_061 = _score_labeled_loo_061(primary_config, primary_checkpoint)
    labeled_067 = _score_labeled_loo_067(secondary_config)
    labeled_merged = labeled_061.merge(labeled_067, on=["id", "label"], how="inner", validate="one_to_one")

    best_params, search_records = select_best_rescue_only_params(
        labels=labeled_merged["label"].to_numpy(dtype="int64"),
        anchor_predictions=labeled_merged["pred_061"].to_numpy(dtype="int64"),
        anchor_argmax=labeled_merged["pred_061_argmax"].to_numpy(dtype="int64"),
        anchor_scores=labeled_merged["score_061_final"].to_numpy(dtype="float32"),
        secondary_predictions=labeled_merged["pred_067"].to_numpy(dtype="int64"),
        secondary_excess_jesse=labeled_merged["score_067_excess_jesse"].to_numpy(dtype="float32"),
        secondary_excess_mila=labeled_merged["score_067_excess_mila"].to_numpy(dtype="float32"),
        other_label=0,
        low_conf_threshold_values=args.low_conf_threshold_values,
        rescue_margin_jesse_values=args.rescue_margin_jesse_values,
        rescue_margin_mila_values=args.rescue_margin_mila_values,
        require_anchor_argmax_match_options=[bool(x) for x in args.require_anchor_argmax_match_options],
    )

    labeled_fused = apply_rescue_only_fusion(
        anchor_predictions=labeled_merged["pred_061"].to_numpy(dtype="int64"),
        anchor_argmax=labeled_merged["pred_061_argmax"].to_numpy(dtype="int64"),
        anchor_scores=labeled_merged["score_061_final"].to_numpy(dtype="float32"),
        secondary_predictions=labeled_merged["pred_067"].to_numpy(dtype="int64"),
        secondary_excess_jesse=labeled_merged["score_067_excess_jesse"].to_numpy(dtype="float32"),
        secondary_excess_mila=labeled_merged["score_067_excess_mila"].to_numpy(dtype="float32"),
        other_label=0,
        low_conf_threshold=float(best_params["low_conf_threshold"]),
        rescue_margin_jesse=float(best_params["rescue_margin_jesse"]),
        rescue_margin_mila=float(best_params["rescue_margin_mila"]),
        require_anchor_argmax_match=bool(best_params["require_anchor_argmax_match"]),
    )
    labeled_merged["pred_fused"] = labeled_fused

    test_061, _, metrics_061 = None, None, None
    val_061, test_061, metrics_061 = _score_neighborhood_tables(primary_config, primary_checkpoint)
    val_067, test_067, metrics_067 = _score_dual_verifier_tables(secondary_config)
    _ = val_061, val_067  # keep naming symmetry; tuning is based on labeled LOO, not fixed val
    test_merged = test_061.merge(test_067, on="id", how="inner", validate="one_to_one")

    test_fused = apply_rescue_only_fusion(
        anchor_predictions=test_merged["pred_061"].to_numpy(dtype="int64"),
        anchor_argmax=test_merged["pred_061_argmax"].to_numpy(dtype="int64"),
        anchor_scores=test_merged["score_061_final"].to_numpy(dtype="float32"),
        secondary_predictions=test_merged["pred_067"].to_numpy(dtype="int64"),
        secondary_excess_jesse=test_merged["score_067_excess_jesse"].to_numpy(dtype="float32"),
        secondary_excess_mila=test_merged["score_067_excess_mila"].to_numpy(dtype="float32"),
        other_label=0,
        low_conf_threshold=float(best_params["low_conf_threshold"]),
        rescue_margin_jesse=float(best_params["rescue_margin_jesse"]),
        rescue_margin_mila=float(best_params["rescue_margin_mila"]),
        require_anchor_argmax_match=bool(best_params["require_anchor_argmax_match"]),
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
    pred_fused_series = test_merged.set_index("id")["pred_fused"].astype(int)

    diff_vs_061 = _prediction_diff_summary(ref_061, pred_fused_series)
    diff_vs_067 = _prediction_diff_summary(ref_067, pred_fused_series)
    diff_vs_082 = _prediction_diff_summary(ref_082, pred_fused_series)

    rescue_mask_labeled = (labeled_merged["pred_061"] == 0) & (labeled_merged["pred_067"] != 0)
    rescue_mask_test = (test_merged["pred_061"] == 0) & (test_merged["pred_067"] != 0)

    metrics = {
        "mode": "rescue_only_classaware_labeled_loo",
        "primary_experiment": primary_config["experiment_name"],
        "secondary_experiment": secondary_config["experiment_name"],
        "primary_checkpoint": str(primary_checkpoint),
        **metrics_061,
        **metrics_067,
        "best_params": best_params,
        "labeled_loo_accuracy_061": float((labeled_merged["pred_061"].to_numpy(dtype="int64") == labeled_merged["label"].to_numpy(dtype="int64")).mean()),
        "labeled_loo_accuracy_067": float((labeled_merged["pred_067"].to_numpy(dtype="int64") == labeled_merged["label"].to_numpy(dtype="int64")).mean()),
        "labeled_loo_accuracy_fused": float((labeled_fused == labeled_merged["label"].to_numpy(dtype="int64")).mean()),
        "labeled_loo_rescue_subset_size": int(rescue_mask_labeled.sum()),
        "labeled_loo_num_changed_vs_061": int((labeled_fused != labeled_merged["pred_061"].to_numpy(dtype="int64")).sum()),
        "test_rescue_subset_size": int(rescue_mask_test.sum()),
        "num_test_changed_vs_061": int((test_fused != test_merged["pred_061"].to_numpy(dtype="int64")).sum()),
        "num_test_changed_vs_082": int((test_fused != ref_082.reindex(test_merged['id']).to_numpy(dtype='int64')).sum()),
        "fused_diff_vs_reference_061": diff_vs_061,
        "fused_diff_vs_reference_067": diff_vs_067,
        "fused_diff_vs_reference_082": diff_vs_082,
        "search_space_size": len(search_records),
        "submission_path": str(submission_path),
    }
    (output_root / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": exp_name,
            "stage": "predict",
            "notes": "scripts/predict_061_067_rescueonly_labeledloo.py::labeled_loo_grid",
            "submission_path": str(submission_path),
            "checkpoint_path": f"{primary_checkpoint} | insightface_buffalo_l_rescueonly",
            "config_path": f"{args.primary_config} | {args.secondary_config}",
        },
    )

    print(json.dumps(metrics, indent=2))
    print(f"submission 已生成: {submission_path}")


if __name__ == "__main__":
    main()
