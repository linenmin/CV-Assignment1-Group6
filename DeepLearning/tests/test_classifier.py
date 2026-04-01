import unittest
from unittest.mock import patch

import torch
from torch import nn

from dl_pipeline.models.classifier import build_classifier


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


class ClassifierFactoryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
