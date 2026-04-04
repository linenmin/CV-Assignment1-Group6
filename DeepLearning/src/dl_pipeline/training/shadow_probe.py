from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from dl_pipeline.data.dataset import FaceDataset
from dl_pipeline.data.transforms import build_eval_transform
from dl_pipeline.inference.prototype import (
    compute_class_prototypes,
    compute_similarity_matrix,
    neighborhood_aware_predictions,
    predict_open_set,
    select_best_neighborhood_threshold,
    select_best_threshold,
)
from dl_pipeline.inference.tta import extract_tta_features


@dataclass(frozen=True)
class ShadowProbeConfig:
    probe_ids: list[int]
    inference_mode: str
    prototype_labels: list[int]
    other_label: int
    config_threshold: float
    threshold_values: list[float]
    csv_path: Path
    neighborhood_top_k: int | None = None
    neighborhood_base_weight: float | None = None
    use_horizontal_flip_tta: bool = False


def build_shadow_probe_frame(test_df: pd.DataFrame, probe_ids: list[int]) -> pd.DataFrame:
    if not probe_ids:
        raise ValueError("probe_ids 不能为空。")

    id_to_rank = {int(probe_id): rank for rank, probe_id in enumerate(probe_ids)}
    selected = test_df[test_df["id"].isin(id_to_rank)].copy()
    missing_ids = [probe_id for probe_id in probe_ids if probe_id not in set(selected["id"].tolist())]
    if missing_ids:
        raise ValueError(f"test.csv 缺少 shadow probe ids: {missing_ids}")

    selected["probe_rank"] = selected["id"].map(id_to_rank)
    selected = selected.sort_values("probe_rank", kind="mergesort").reset_index(drop=True)
    return selected.drop(columns=["probe_rank"])


def resolve_shadow_probe_config(config: dict, output_root: Path) -> ShadowProbeConfig | None:
    shadow_probe_cfg = config.get("shadow_probe", {})
    if not shadow_probe_cfg or not shadow_probe_cfg.get("enabled", False):
        return None

    inference_cfg = config.get("inference", {})
    inference_mode = str(inference_cfg.get("mode", "prototype"))
    threshold = float(inference_cfg.get("threshold", 0.55))
    threshold_values = shadow_probe_cfg.get("threshold_values") or inference_cfg.get("threshold_values") or [threshold]
    neighborhood_cfg = inference_cfg.get("neighborhood_aware", {})
    csv_name = str(shadow_probe_cfg.get("csv_name", "shadow_probe_history.csv"))

    return ShadowProbeConfig(
        probe_ids=[int(probe_id) for probe_id in shadow_probe_cfg.get("probe_ids", [])],
        inference_mode=inference_mode,
        prototype_labels=[int(label) for label in inference_cfg.get("prototype_labels", [1, 2])],
        other_label=int(inference_cfg.get("other_label", 0)),
        config_threshold=threshold,
        threshold_values=[float(value) for value in threshold_values],
        csv_path=output_root / csv_name,
        neighborhood_top_k=int(neighborhood_cfg.get("top_k", 15)) if inference_mode == "neighborhood_aware" else None,
        neighborhood_base_weight=(
            float(neighborhood_cfg.get("base_weight", 0.5))
            if inference_mode == "neighborhood_aware"
            else None
        ),
        use_horizontal_flip_tta=bool(inference_cfg.get("tta_horizontal_flip", False)),
    )


def compute_shadow_probe_epoch_records(
    *,
    epoch: int,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_ids: torch.Tensor,
    probe_ids: list[int],
    inference_mode: str,
    prototype_labels: list[int],
    other_label: int,
    config_threshold: float,
    threshold_values: list[float],
    neighborhood_top_k: int | None = None,
    neighborhood_base_weight: float | None = None,
) -> list[dict[str, float | int]]:
    prototypes = compute_class_prototypes(train_embeddings, train_labels, prototype_labels=prototype_labels)
    similarity_matrix, ordered_labels = compute_similarity_matrix(
        test_embeddings,
        prototypes,
        prototype_labels=prototype_labels,
    )
    best_indices = similarity_matrix.argmax(dim=1)
    argmax_labels = torch.tensor(
        [ordered_labels[index] for index in best_indices.tolist()],
        device=test_embeddings.device,
        dtype=torch.long,
    )

    if inference_mode == "neighborhood_aware":
        if neighborhood_top_k is None or neighborhood_base_weight is None:
            raise ValueError("neighborhood_aware shadow probe 缺少 top_k/base_weight。")
        pred_config, base_scores, neighbor_scores, final_scores = neighborhood_aware_predictions(
            query_embeddings=test_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=float(config_threshold),
            top_k=int(neighborhood_top_k),
            base_weight=float(neighborhood_base_weight),
        )
        selected_threshold, selected_accuracy, _ = select_best_neighborhood_threshold(
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            prototypes=prototypes,
            other_label=other_label,
            threshold_values=[float(value) for value in threshold_values],
            top_k=int(neighborhood_top_k),
            base_weight=float(neighborhood_base_weight),
        )
        pred_selected, _, _, _ = neighborhood_aware_predictions(
            query_embeddings=test_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=float(selected_threshold),
            top_k=int(neighborhood_top_k),
            base_weight=float(neighborhood_base_weight),
        )
    elif inference_mode == "prototype":
        pred_config, base_scores = predict_open_set(
            query_embeddings=test_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=float(config_threshold),
        )
        selected_threshold, selected_accuracy = select_best_threshold(
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            prototypes=prototypes,
            other_label=other_label,
            threshold_values=[float(value) for value in threshold_values],
        )
        pred_selected, _ = predict_open_set(
            query_embeddings=test_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=float(selected_threshold),
        )
        neighbor_scores = torch.full_like(base_scores, float("nan"))
        final_scores = base_scores
    else:
        raise ValueError(f"当前 shadow probe 不支持 inference.mode={inference_mode}")

    id_to_index = {int(sample_id): index for index, sample_id in enumerate(test_ids.tolist())}
    records: list[dict[str, float | int]] = []
    for probe_id in probe_ids:
        if int(probe_id) not in id_to_index:
            raise ValueError(f"test embeddings 缺少 shadow probe id={probe_id}")
        row_index = id_to_index[int(probe_id)]
        record: dict[str, float | int] = {
            "epoch": int(epoch),
            "id": int(probe_id),
            "argmax_label": int(argmax_labels[row_index].item()),
            "pred_config_threshold": int(pred_config[row_index].item()),
            "pred_val_selected_threshold": int(pred_selected[row_index].item()),
            "config_threshold": float(config_threshold),
            "selected_threshold": float(selected_threshold),
            "selected_threshold_val_accuracy": float(selected_accuracy),
            "base_score": float(base_scores[row_index].item()),
            "neighbor_score": float(neighbor_scores[row_index].item()),
            "final_score": float(final_scores[row_index].item()),
        }
        for label_index, label in enumerate(ordered_labels):
            record[f"score_class{int(label)}"] = float(similarity_matrix[row_index, label_index].item())
        records.append(record)
    return records


class ShadowProbeEpochCallback(Callback):
    def __init__(self, probe_config: ShadowProbeConfig, image_size: int, normalization: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.probe_config = probe_config
        self.image_size = int(image_size)
        self.normalization = normalization
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self._history: list[dict[str, object]] = []
        self._eval_train_loader: DataLoader | None = None
        self._eval_val_loader: DataLoader | None = None
        self._eval_test_loader: DataLoader | None = None
        self._probe_frame: pd.DataFrame | None = None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            return
        datamodule = trainer.datamodule
        if datamodule is None:
            raise ValueError("shadow probe 需要 datamodule。")

        eval_transform = build_eval_transform(self.image_size, normalization=self.normalization)
        train_frame = datamodule.train_df
        if bool(getattr(datamodule, "use_full_train", False)):
            train_frame = (
                pd.concat([datamodule.train_df, datamodule.val_df], ignore_index=True)
                .sort_values("id", kind="mergesort")
                .reset_index(drop=True)
            )

        self._eval_train_loader = DataLoader(
            FaceDataset(train_frame, eval_transform, True),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._eval_val_loader = DataLoader(
            FaceDataset(datamodule.val_df, eval_transform, True),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._eval_test_loader = DataLoader(
            FaceDataset(datamodule.test_df, eval_transform, False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._probe_frame = build_shadow_probe_frame(datamodule.test_df, self.probe_config.probe_ids)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return
        if self._eval_train_loader is None or self._eval_val_loader is None or self._eval_test_loader is None:
            raise ValueError("shadow probe callback 尚未完成 setup。")

        try:
            train_embeddings, train_labels = self._collect_embeddings(
                pl_module,
                self._eval_train_loader,
                use_horizontal_flip_tta=self.probe_config.use_horizontal_flip_tta,
            )
            val_embeddings, val_labels = self._collect_embeddings(
                pl_module,
                self._eval_val_loader,
                use_horizontal_flip_tta=self.probe_config.use_horizontal_flip_tta,
            )
            test_embeddings, test_ids = self._collect_embeddings(
                pl_module,
                self._eval_test_loader,
                use_horizontal_flip_tta=self.probe_config.use_horizontal_flip_tta,
            )

            records = compute_shadow_probe_epoch_records(
                epoch=int(trainer.current_epoch) + 1,
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                val_embeddings=val_embeddings,
                val_labels=val_labels,
                test_embeddings=test_embeddings,
                test_ids=test_ids,
                probe_ids=self.probe_config.probe_ids,
                inference_mode=self.probe_config.inference_mode,
                prototype_labels=self.probe_config.prototype_labels,
                other_label=self.probe_config.other_label,
                config_threshold=self.probe_config.config_threshold,
                threshold_values=self.probe_config.threshold_values,
                neighborhood_top_k=self.probe_config.neighborhood_top_k,
                neighborhood_base_weight=self.probe_config.neighborhood_base_weight,
            )
            if self._probe_frame is not None:
                probe_meta = self._probe_frame[["id", "image_path", "source_path"]].copy()
                merged_records = pd.DataFrame(records).merge(probe_meta, on="id", how="left")
                self._history.extend(merged_records.to_dict(orient="records"))
            else:
                self._history.extend(records)

            history_df = pd.DataFrame(self._history)
            history_df.to_csv(self.probe_config.csv_path, index=False, encoding="utf-8")
        finally:
            # Lightning 在验证阶段会切到 eval；这里额外做过 probe 导出，结束后显式切回 train，
            # 避免后续 epoch 继续停留在 eval 模式。
            pl_module.train()

    def _collect_embeddings(
        self,
        pl_module: L.LightningModule,
        dataloader: DataLoader,
        *,
        use_horizontal_flip_tta: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        values = []
        pl_module.eval()
        with torch.no_grad():
            for images, batch_values in dataloader:
                features = extract_tta_features(
                    model=pl_module,
                    images=images,
                    device=pl_module.device,
                    use_horizontal_flip=use_horizontal_flip_tta,
                )
                embeddings.append(features.detach().cpu())
                values.append(batch_values.detach().cpu())
        return torch.cat(embeddings, dim=0), torch.cat(values, dim=0)
