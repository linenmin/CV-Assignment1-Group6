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
    def test_cosface_training_step_finite_loss(self, mock_build_classifier):
        mock_build_classifier.return_value = DummyModel()

        module = FaceClassifierModule(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            learning_rate=1e-4,
            backbone_learning_rate=1e-4,
            weight_decay=1e-4,
            scheduler_name="cosine",
            max_epochs=10,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=False,
            unfreeze_last_stage=False,
            loss_name="cosface",
            cosface_scale=16.0,
            cosface_margin=0.2,
        )

        images = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 2, 0])
        loss = module.training_step((images, labels), 0)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    @patch("dl_pipeline.training.lightning_module.build_classifier")
    def test_bce_ovr_training_step_uses_other_as_shared_negative_and_returns_finite_loss(
        self, mock_build_classifier
    ):
        mock_build_classifier.return_value = DummyModel()

        module = FaceClassifierModule(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            learning_rate=1e-4,
            backbone_learning_rate=1e-4,
            weight_decay=1e-4,
            scheduler_name="cosine",
            max_epochs=10,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=False,
            unfreeze_last_stage=False,
            loss_name="bce_ovr",
            loss_target_labels=[1, 2],
        )

        images = torch.randn(5, 4)
        labels = torch.tensor([0, 1, 2, 0, 1])
        loss = module.training_step((images, labels), 0)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    @patch("dl_pipeline.training.lightning_module.build_classifier")
    def test_bce_ovr_validation_rejects_low_confidence_examples_to_other(self, mock_build_classifier):
        mock_build_classifier.return_value = DummyModel()

        module = FaceClassifierModule(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            learning_rate=1e-4,
            backbone_learning_rate=1e-4,
            weight_decay=1e-4,
            scheduler_name="cosine",
            max_epochs=10,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=False,
            unfreeze_last_stage=False,
            loss_name="bce_ovr",
            loss_target_labels=[1, 2],
        )

        images = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1, 2])

        with torch.no_grad():
            module.model.backbone.weight.copy_(torch.eye(4))
            module.model.backbone.bias.zero_()
            module.ovr_head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
            module.ovr_head.bias.copy_(torch.tensor([-2.0, -2.0]))

        loss, preds, metric_labels, batch_size = module._compute_loss_and_predictions(images, labels)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(batch_size, 3)
        self.assertEqual(metric_labels.tolist(), [0, 1, 2])
        self.assertEqual(preds.tolist(), [0, 1, 2])

    @patch("dl_pipeline.training.lightning_module.build_classifier")
    def test_bce_ovr_supcon_training_step_returns_finite_loss_with_target_positive_pairs(
        self, mock_build_classifier
    ):
        mock_build_classifier.return_value = DummyModel()

        module = FaceClassifierModule(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            learning_rate=1e-4,
            backbone_learning_rate=1e-4,
            weight_decay=1e-4,
            scheduler_name="cosine",
            max_epochs=10,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=False,
            unfreeze_last_stage=False,
            loss_name="bce_ovr_supcon",
            loss_target_labels=[1, 2],
            supcon_weight=0.1,
            supcon_temperature=0.1,
        )

        images = torch.randn(6, 4)
        labels = torch.tensor([1, 1, 2, 2, 0, 0])
        loss = module.training_step((images, labels), 0)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    @patch("dl_pipeline.training.lightning_module.build_classifier")
    def test_bce_ovr_supcon_aux_loss_is_zero_when_no_target_positive_pairs(self, mock_build_classifier):
        mock_build_classifier.return_value = DummyModel()

        module = FaceClassifierModule(
            model_family="cvlface",
            backbone_name="ir101",
            num_classes=3,
            pretrained=True,
            dropout=0.2,
            learning_rate=1e-4,
            backbone_learning_rate=1e-4,
            weight_decay=1e-4,
            scheduler_name="cosine",
            max_epochs=10,
            pretrained_repo_id="minchul/cvlface_adaface_ir101_webface4m",
            freeze_backbone=False,
            unfreeze_last_stage=False,
            loss_name="bce_ovr_supcon",
            loss_target_labels=[1, 2],
            supcon_weight=0.1,
            supcon_temperature=0.1,
        )

        features = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 2, 0])
        supcon_loss = module._compute_target_only_supcon_loss(features, labels)

        self.assertTrue(torch.isfinite(supcon_loss))
        self.assertEqual(float(supcon_loss.item()), 0.0)

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
