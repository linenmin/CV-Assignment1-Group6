import unittest
from unittest.mock import patch

import torch
from torch import nn

from dl_pipeline.training.lightning_module import FaceClassifierModule


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 3)

    def forward(self, images):
        return self.classifier(self.backbone(images))

    def extract_features(self, images):
        return self.backbone(images)


class LightningModuleOptimizerTests(unittest.TestCase):
    @patch("dl_pipeline.training.lightning_module.build_classifier")
    def test_arcface_training_step_ignores_other_class_and_returns_finite_loss(self, mock_build_classifier):
        mock_build_classifier.return_value = DummyModel()

        module = FaceClassifierModule(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            learning_rate=3e-4,
            backbone_learning_rate=3e-5,
            weight_decay=1e-4,
            scheduler_name="cosine",
            max_epochs=10,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=False,
            unfreeze_last_stage=False,
            loss_name="arcface",
            loss_target_labels=[1, 2],
            arcface_scale=16.0,
            arcface_margin=0.2,
        )

        images = torch.randn(3, 4)
        labels = torch.tensor([0, 1, 2])
        loss = module.training_step((images, labels), 0)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    @patch("dl_pipeline.training.lightning_module.build_classifier")
    def test_configure_optimizers_separates_backbone_learning_rate(self, mock_build_classifier):
        mock_build_classifier.return_value = DummyModel()

        module = FaceClassifierModule(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            learning_rate=3e-4,
            backbone_learning_rate=3e-5,
            weight_decay=1e-4,
            scheduler_name="cosine",
            max_epochs=10,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=False,
            unfreeze_last_stage=False,
        )

        optimizer_bundle = module.configure_optimizers()
        optimizer = optimizer_bundle["optimizer"]

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 3e-5)
        self.assertEqual(optimizer.param_groups[1]["lr"], 3e-4)

    @patch("dl_pipeline.training.lightning_module.build_classifier")
    def test_configure_optimizers_uses_custom_monitor_for_plateau(self, mock_build_classifier):
        mock_build_classifier.return_value = DummyModel()

        module = FaceClassifierModule(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            learning_rate=3e-4,
            backbone_learning_rate=3e-5,
            weight_decay=1e-4,
            scheduler_name="plateau",
            max_epochs=10,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=False,
            unfreeze_last_stage=False,
            monitor_metric="val_loss",
        )

        optimizer_bundle = module.configure_optimizers()

        self.assertEqual(optimizer_bundle["lr_scheduler"]["monitor"], "val_loss")


if __name__ == "__main__":
    unittest.main()
