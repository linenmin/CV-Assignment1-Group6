import unittest
from unittest.mock import patch

import torch
from torch import nn

from dl_pipeline.models.classifier import ArcFaceHead, build_classifier


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 512)

    def forward(self, x):
        return self.proj(x)


class DummyCVLFaceBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.net = nn.Module()
        self.model.net.input_layer = nn.Linear(4, 4)
        self.model.net.body = nn.Sequential(*[nn.Linear(4, 4) for _ in range(49)])
        self.model.net.output_layer = nn.Linear(4, 512)

    def forward(self, x):
        return torch.randn(x.shape[0], 512)


class DummyCVLFaceViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.net = nn.Module()
        self.model.net.patch_embed = nn.Linear(4, 4)
        self.model.net.pos_drop = nn.Dropout(0.0)
        self.model.net.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(24)])
        self.model.net.norm = nn.LayerNorm(4)
        self.model.net.feature = nn.Sequential(nn.Linear(4, 512))

    def forward(self, x):
        return torch.randn(x.shape[0], 512)


class ClassifierFactoryTests(unittest.TestCase):
    def test_arcface_head_applies_margin_to_target_class_only(self):
        head = ArcFaceHead(in_features=2, num_classes=2, scale=1.0, margin=0.5)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        features = torch.tensor([[1.0, 0.0]])
        labels = torch.tensor([0])

        logits_without_margin = head(features)
        logits_with_margin = head(features, labels)

        self.assertLess(logits_with_margin[0, 0].item(), logits_without_margin[0, 0].item())
        self.assertAlmostEqual(
            logits_with_margin[0, 1].item(),
            logits_without_margin[0, 1].item(),
            places=6,
        )

    @patch("dl_pipeline.models.classifier.load_cvlface_backbone")
    def test_build_classifier_can_wrap_cvlface_backbone(self, mock_loader):
        mock_loader.return_value = DummyBackbone()

        model = build_classifier(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=True,
        )

        self.assertEqual(model.classifier[-1].out_features, 3)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))

        logits = model(torch.randn(2, 4))
        self.assertEqual(tuple(logits.shape), (2, 3))

        features = model.extract_features(torch.randn(2, 4))
        self.assertEqual(tuple(features.shape), (2, 512))

    @patch("dl_pipeline.models.classifier.load_cvlface_backbone")
    def test_build_classifier_can_unfreeze_only_last_cvlface_stage(self, mock_loader):
        mock_loader.return_value = DummyCVLFaceBackbone()

        model = build_classifier(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=True,
            unfreeze_last_stage=True,
        )

        frozen_block = model.backbone.model.net.body[45]
        trainable_block = model.backbone.model.net.body[48]

        self.assertTrue(all(not p.requires_grad for p in model.backbone.model.net.input_layer.parameters()))
        self.assertTrue(all(not p.requires_grad for p in frozen_block.parameters()))
        self.assertTrue(all(p.requires_grad for p in trainable_block.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.model.net.output_layer.parameters()))

    @patch("dl_pipeline.models.classifier.load_cvlface_backbone")
    def test_build_classifier_can_unfreeze_last_two_cvlface_stages(self, mock_loader):
        mock_loader.return_value = DummyCVLFaceBackbone()

        model = build_classifier(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=True,
            unfreeze_stage_count=2,
        )

        frozen_block = model.backbone.model.net.body[15]
        trainable_block = model.backbone.model.net.body[16]

        self.assertTrue(all(not p.requires_grad for p in model.backbone.model.net.input_layer.parameters()))
        self.assertTrue(all(not p.requires_grad for p in frozen_block.parameters()))
        self.assertTrue(all(p.requires_grad for p in trainable_block.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.model.net.output_layer.parameters()))

    @patch("dl_pipeline.models.classifier.load_cvlface_backbone")
    def test_build_classifier_can_unfreeze_last_two_vit_blocks(self, mock_loader):
        mock_loader.return_value = DummyCVLFaceViTBackbone()

        model = build_classifier(
            model_family="cvlface",
            backbone_name="vit_base",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            pretrained_repo_id="minchul/cvlface_adaface_vit_base_webface4m",
            freeze_backbone=True,
            unfreeze_stage_count=2,
        )

        frozen_block = model.backbone.model.net.blocks[21]
        trainable_blocks = model.backbone.model.net.blocks[22:]

        self.assertTrue(all(not p.requires_grad for p in model.backbone.model.net.patch_embed.parameters()))
        self.assertTrue(all(not p.requires_grad for p in frozen_block.parameters()))
        for block in trainable_blocks:
            self.assertTrue(all(p.requires_grad for p in block.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.model.net.norm.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.model.net.feature.parameters()))

    @patch("dl_pipeline.models.classifier.load_cvlface_backbone")
    def test_build_classifier_can_freeze_vit_feature_and_norm_during_partial_unfreeze(self, mock_loader):
        mock_loader.return_value = DummyCVLFaceViTBackbone()

        model = build_classifier(
            model_family="cvlface",
            backbone_name="vit_base",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            pretrained_repo_id="minchul/cvlface_adaface_vit_base_webface4m",
            freeze_backbone=True,
            unfreeze_stage_count=2,
            unfreeze_cvlface_norm=False,
            unfreeze_cvlface_feature=False,
        )

        frozen_block = model.backbone.model.net.blocks[21]
        trainable_blocks = model.backbone.model.net.blocks[22:]

        self.assertTrue(all(not p.requires_grad for p in model.backbone.model.net.patch_embed.parameters()))
        self.assertTrue(all(not p.requires_grad for p in frozen_block.parameters()))
        for block in trainable_blocks:
            self.assertTrue(all(p.requires_grad for p in block.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.model.net.norm.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.model.net.feature.parameters()))

    @patch("dl_pipeline.models.classifier.load_cvlface_backbone")
    def test_build_classifier_rejects_too_many_vit_blocks_to_unfreeze(self, mock_loader):
        mock_loader.return_value = DummyCVLFaceViTBackbone()

        with self.assertRaises(ValueError):
            build_classifier(
                model_family="cvlface",
                backbone_name="vit_base",
                num_classes=3,
                pretrained=True,
                dropout=0.2,
                pretrained_repo_id="minchul/cvlface_adaface_vit_base_webface4m",
                freeze_backbone=True,
                unfreeze_stage_count=25,
            )


if __name__ == "__main__":
    unittest.main()
