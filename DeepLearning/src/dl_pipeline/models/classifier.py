from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_pipeline.models.cvlface import load_cvlface_backbone


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _resolve_cvlface_stage_lengths(backbone: nn.Module) -> list[int]:
    try:
        net = backbone.model.net
        body = net.body
    except AttributeError as exc:
        raise ValueError("当前 cvlface backbone 结构不支持按 stage 解冻。") from exc

    stage_lengths_by_body_size = {
        8: [2, 2, 2, 2],
        16: [3, 4, 6, 3],
        24: [3, 4, 14, 3],
        49: [3, 13, 30, 3],
    }
    stage_lengths = stage_lengths_by_body_size.get(len(body))
    if stage_lengths is None:
        raise ValueError(f"未知的 cvlface body 长度: {len(body)}")
    return stage_lengths


def _unfreeze_last_cvlface_stages(backbone: nn.Module, stage_count: int) -> None:
    if stage_count <= 0:
        return

    stage_lengths = _resolve_cvlface_stage_lengths(backbone)
    if stage_count > len(stage_lengths):
        raise ValueError(f"请求解冻 {stage_count} 个 stage，但当前 backbone 只有 {len(stage_lengths)} 个 stage。")

    net = backbone.model.net
    body = net.body
    trainable_block_count = sum(stage_lengths[-stage_count:])
    _set_requires_grad(net.output_layer, True)
    for block in body[-trainable_block_count:]:
        _set_requires_grad(block, True)


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        normalized_features = F.normalize(features, p=2, dim=1)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(normalized_features, normalized_weight).clamp(-1.0, 1.0)
        if labels is None:
            return cosine * self.scale

        if labels.ndim != 1:
            raise ValueError("ArcFace labels 必须是一维张量。")
        if labels.shape[0] != cosine.shape[0]:
            raise ValueError("ArcFace labels 数量必须与 features batch 大小一致。")

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(0.0))
        phi = cosine * self.cos_margin - sine * self.sin_margin
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=cosine.dtype, device=cosine.device)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale


class CVLFaceClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float,
        freeze_backbone: bool,
        unfreeze_last_stage: bool = False,
        unfreeze_stage_count: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        effective_unfreeze_stage_count = max(unfreeze_stage_count, 1 if unfreeze_last_stage else 0)
        if freeze_backbone or effective_unfreeze_stage_count > 0:
            _set_requires_grad(self.backbone, False)
        if effective_unfreeze_stage_count > 0:
            _unfreeze_last_cvlface_stages(self.backbone, effective_unfreeze_stage_count)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def extract_features(self, x):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        return features

    def forward(self, x):
        return self.classifier(self.extract_features(x))


def build_classifier(
    model_family: str,
    backbone_name: str,
    num_classes: int,
    pretrained: bool,
    dropout: float,
    pretrained_repo_id: str | None = None,
    freeze_backbone: bool = False,
    unfreeze_last_stage: bool = False,
    unfreeze_stage_count: int = 0,
) -> nn.Module:
    if model_family == "cvlface":
        if not pretrained_repo_id:
            raise ValueError("cvlface 模型需要提供 pretrained_repo_id。")
        backbone = load_cvlface_backbone(pretrained_repo_id)
        return CVLFaceClassifier(
            backbone=backbone,
            feature_dim=512,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            unfreeze_last_stage=unfreeze_last_stage,
            unfreeze_stage_count=unfreeze_stage_count,
        )

    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
