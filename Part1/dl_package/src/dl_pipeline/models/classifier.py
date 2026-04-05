from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_pipeline.models.cvlface import load_cvlface_backbone
from dl_pipeline.models.lvface import load_lvface_backbone, resolve_pretrained_checkpoint
from dl_pipeline.models.iresnet import iresnet50 as arcface_iresnet50


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


def _resolve_cvlface_vit_block_count(backbone: nn.Module) -> int:
    try:
        net = backbone.model.net
        blocks = net.blocks
    except AttributeError as exc:
        raise ValueError("当前 cvlface ViT backbone 结构不支持按 block 解冻。") from exc

    return len(blocks)


def _unfreeze_last_cvlface_stages(
    backbone: nn.Module,
    stage_count: int,
    *,
    unfreeze_vit_norm: bool = True,
    unfreeze_vit_feature: bool = True,
) -> None:
    if stage_count <= 0:
        return

    net = backbone.model.net
    if hasattr(net, "body"):
        stage_lengths = _resolve_cvlface_stage_lengths(backbone)
        if stage_count > len(stage_lengths):
            raise ValueError(f"请求解冻 {stage_count} 个 stage，但当前 backbone 只有 {len(stage_lengths)} 个 stage。")

        body = net.body
        trainable_block_count = sum(stage_lengths[-stage_count:])
        _set_requires_grad(net.output_layer, True)
        for block in body[-trainable_block_count:]:
            _set_requires_grad(block, True)
        return

    if hasattr(net, "blocks"):
        block_count = _resolve_cvlface_vit_block_count(backbone)
        if stage_count > block_count:
            raise ValueError(f"请求解冻 {stage_count} 个 block，但当前 backbone 只有 {block_count} 个 block。")

        for block in net.blocks[-stage_count:]:
            _set_requires_grad(block, True)
        if unfreeze_vit_norm and hasattr(net, "norm"):
            _set_requires_grad(net.norm, True)
        if unfreeze_vit_feature and hasattr(net, "feature"):
            _set_requires_grad(net.feature, True)
        return

    raise ValueError("当前 cvlface backbone 结构不支持按 stage/block 解冻。")


def _unfreeze_last_lvface_vit_blocks(
    backbone: nn.Module,
    block_count: int,
    *,
    unfreeze_norm: bool = True,
    unfreeze_feature: bool = True,
) -> None:
    """LVFace VisionTransformer：按最后若干 Block 解冻，可选解冻 ``norm`` / ``feature``。"""
    if block_count <= 0:
        return
    if not hasattr(backbone, "blocks"):
        raise ValueError(
            "当前 lvface backbone 不是 VisionTransformer（缺少 blocks）。"
            "若使用 ResNet 系 LVFace，请暂时将 unfreeze_stage_count 设为 0，或扩展解冻逻辑。"
        )
    blocks = backbone.blocks
    if block_count > len(blocks):
        raise ValueError(
            f"请求解冻 {block_count} 个 block，但当前 backbone 只有 {len(blocks)} 个 block。"
        )
    for block in blocks[-block_count:]:
        _set_requires_grad(block, True)
    if unfreeze_norm and hasattr(backbone, "norm"):
        _set_requires_grad(backbone.norm, True)
    if unfreeze_feature and hasattr(backbone, "feature"):
        _set_requires_grad(backbone.feature, True)


def _freeze_iresnet_backbone_bn_only(backbone: nn.Module) -> None:
    """冻结 Conv/PReLU/Linear，仅 BatchNorm2d / BatchNorm1d 可训练（域适应常用）。"""
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for module in backbone.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for parameter in module.parameters():
                parameter.requires_grad = True


def _configure_arcface_iresnet_trainable(backbone: nn.Module, mode: str = "bn_only") -> None:
    """控制 InsightFace IResNet backbone 的可训练范围。

    - ``bn_only``：仅 BN（与原行为一致）。
    - ``last_stage``：冻结 stem/layer1–3，训练 layer4、bn2、fc、features（经典 ImageNet 式「训最后一组 stage」）。
    - ``last_two_stages``：在 ``last_stage`` 基础上再解冻 layer3（数据较多时可试）。
    """
    if mode == "bn_only":
        _freeze_iresnet_backbone_bn_only(backbone)
        return
    if mode not in ("last_stage", "last_two_stages"):
        raise ValueError(
            f"iresnet_finetune_mode 应为 bn_only | last_stage | last_two_stages，得到 {mode!r}。"
        )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if mode == "last_two_stages":
        for parameter in backbone.layer3.parameters():
            parameter.requires_grad = True
    for parameter in backbone.layer4.parameters():
        parameter.requires_grad = True
    for parameter in backbone.bn2.parameters():
        parameter.requires_grad = True
    for parameter in backbone.fc.parameters():
        parameter.requires_grad = True
    for parameter in backbone.features.parameters():
        parameter.requires_grad = True


class IResNetFaceClassifier(nn.Module):
    """InsightFace arcface_torch IResNet50（512-d）+ 线性分类头。"""

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def imprint_linear_head_from_loader(
    model: nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
) -> None:
    """Weight Imprinting：用训练集上各类 L2 归一化 embedding 的均值（再 L2 归一化）初始化最后一层 Linear 权重。

    适用于 ``IResNetFaceClassifier`` 等结构：``extract_features`` + ``classifier`` 末尾为 ``nn.Linear``。
    见 Park et al. Open-Set Face Identification on Few-Shot Gallery by Fine-Tuning (ICPR 2022).
    """
    if not hasattr(model, "extract_features"):
        raise ValueError("weight imprinting 需要模型实现 extract_features。")
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Sequential) or len(classifier) < 1:
        raise ValueError("weight imprinting 需要 model.classifier 为至少一层的 Sequential。")
    linear = classifier[-1]
    if not isinstance(linear, nn.Linear):
        raise ValueError(f"weight imprinting 需要 classifier 最后一层为 Linear，得到 {type(linear)}。")
    if linear.out_features != num_classes:
        raise ValueError(
            f"Linear.out_features ({linear.out_features}) 与 num_classes ({num_classes}) 不一致。"
        )
    in_features = linear.in_features
    sum_feats = torch.zeros(num_classes, in_features, device=device, dtype=torch.float32)
    counts = torch.zeros(num_classes, device=device, dtype=torch.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feats = model.extract_features(images)
            if feats.ndim != 2 or feats.shape[1] != in_features:
                raise ValueError(
                    f"extract_features 应返回 [N,{in_features}]，得到 shape={tuple(feats.shape)}。"
                )
            feats = F.normalize(feats.float(), p=2, dim=1)
            for c in range(num_classes):
                mask = labels == c
                if mask.any():
                    sum_feats[c] = sum_feats[c] + feats[mask].sum(dim=0)
                    counts[c] = counts[c] + mask.sum().to(dtype=torch.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            proto = sum_feats[c] / counts[c].clamp_min(1.0)
            proto = F.normalize(proto.unsqueeze(0), p=2, dim=1).squeeze(0)
            linear.weight.data[c].copy_(proto)
    if linear.bias is not None:
        linear.bias.data.zero_()
    model.train(was_training)


def imprint_cosine_classifier_weight_from_loader(
    model: nn.Module,
    weight: nn.Parameter,
    data_loader,
    device: torch.device,
    num_classes: int,
) -> None:
    """将各类 L2 归一化 embedding 均值（再归一化）写入 CosFace/ArcFace 等余弦分类头的 ``weight`` 行向量。"""
    if not hasattr(model, "extract_features"):
        raise ValueError("需要模型实现 extract_features。")
    if weight.ndim != 2 or weight.shape[0] != num_classes:
        raise ValueError(f"weight 形状应为 [{num_classes}, D]，得到 {tuple(weight.shape)}。")
    in_features = weight.shape[1]
    sum_feats = torch.zeros(num_classes, in_features, device=device, dtype=torch.float32)
    counts = torch.zeros(num_classes, device=device, dtype=torch.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feats = model.extract_features(images)
            if feats.ndim != 2 or feats.shape[1] != in_features:
                raise ValueError(
                    f"extract_features 应返回 [N,{in_features}]，得到 shape={tuple(feats.shape)}。"
                )
            feats = F.normalize(feats.float(), p=2, dim=1)
            for c in range(num_classes):
                mask = labels == c
                if mask.any():
                    sum_feats[c] = sum_feats[c] + feats[mask].sum(dim=0)
                    counts[c] = counts[c] + mask.sum().to(dtype=torch.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            proto = sum_feats[c] / counts[c].clamp_min(1.0)
            proto = F.normalize(proto.unsqueeze(0), p=2, dim=1).squeeze(0)
            weight.data[c].copy_(proto)
    model.train(was_training)


class CosFaceHead(nn.Module):
    """Large Margin Cosine Loss (CosFace)：logits = s * cos - s * m * one_hot(y)。"""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.35,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        normalized_features = F.normalize(features, p=2, dim=1)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(normalized_features, normalized_weight).clamp(-1.0, 1.0)
        logits = cosine * self.scale
        if labels is None:
            return logits
        if labels.ndim != 1:
            raise ValueError("CosFace labels 必须是一维张量。")
        if labels.shape[0] != cosine.shape[0]:
            raise ValueError("CosFace labels 数量必须与 batch 大小一致。")
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=logits.dtype, device=logits.device)
        return logits - self.scale * self.margin * one_hot


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
        unfreeze_cvlface_norm: bool = True,
        unfreeze_cvlface_feature: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        effective_unfreeze_stage_count = max(unfreeze_stage_count, 1 if unfreeze_last_stage else 0)
        if freeze_backbone or effective_unfreeze_stage_count > 0:
            _set_requires_grad(self.backbone, False)
        if effective_unfreeze_stage_count > 0:
            _unfreeze_last_cvlface_stages(
                self.backbone,
                effective_unfreeze_stage_count,
                unfreeze_vit_norm=unfreeze_cvlface_norm,
                unfreeze_vit_feature=unfreeze_cvlface_feature,
            )
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


class LVFaceClassifier(nn.Module):
    """LVFace 预训练 backbone + 线性分类头；接口与 ``CVLFaceClassifier`` 一致。"""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float,
        freeze_backbone: bool,
        unfreeze_last_stage: bool = False,
        unfreeze_stage_count: int = 0,
        unfreeze_cvlface_norm: bool = True,
        unfreeze_cvlface_feature: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        effective_unfreeze_stage_count = max(unfreeze_stage_count, 1 if unfreeze_last_stage else 0)
        if freeze_backbone or effective_unfreeze_stage_count > 0:
            _set_requires_grad(self.backbone, False)
        if effective_unfreeze_stage_count > 0:
            _unfreeze_last_lvface_vit_blocks(
                self.backbone,
                effective_unfreeze_stage_count,
                unfreeze_norm=unfreeze_cvlface_norm,
                unfreeze_feature=unfreeze_cvlface_feature,
            )
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
    unfreeze_cvlface_norm: bool = True,
    unfreeze_cvlface_feature: bool = True,
    pretrained_checkpoint_path: str | None = None,
    iresnet_finetune_mode: str = "bn_only",
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
            unfreeze_cvlface_norm=unfreeze_cvlface_norm,
            unfreeze_cvlface_feature=unfreeze_cvlface_feature,
        )

    if model_family == "lvface":
        if not pretrained_checkpoint_path:
            raise ValueError("lvface 模型需要提供 pretrained_checkpoint_path（本地 LVFace .pt 权重）。")
        checkpoint_path = resolve_pretrained_checkpoint(pretrained_checkpoint_path)
        backbone = load_lvface_backbone(backbone_name, checkpoint_path)
        return LVFaceClassifier(
            backbone=backbone,
            feature_dim=512,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            unfreeze_last_stage=unfreeze_last_stage,
            unfreeze_stage_count=unfreeze_stage_count,
            unfreeze_cvlface_norm=unfreeze_cvlface_norm,
            unfreeze_cvlface_feature=unfreeze_cvlface_feature,
        )

    if model_family == "arcface_iresnet":
        if not pretrained_checkpoint_path:
            raise ValueError(
                "arcface_iresnet 需要提供 pretrained_checkpoint_path（Glint360K 等 IResNet50 state_dict .pth）。"
            )
        ckpt_path = resolve_pretrained_checkpoint(pretrained_checkpoint_path)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise ValueError(f"预训练文件应为 state_dict，得到: {type(raw)}")
        backbone = arcface_iresnet50(dropout=0.0, fp16=False)
        missing, unexpected = backbone.load_state_dict(raw, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"load_state_dict 异常 missing={missing} unexpected={unexpected}")
        _configure_arcface_iresnet_trainable(backbone, iresnet_finetune_mode)
        return IResNetFaceClassifier(backbone=backbone, num_classes=num_classes, dropout=dropout)

    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
