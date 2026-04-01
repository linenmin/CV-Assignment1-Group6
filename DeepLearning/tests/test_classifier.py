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


if __name__ == "__main__":
    unittest.main()
