import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_pipeline.models.classifier import (
    ArcFaceHead,
    CosFaceHead,
    build_classifier,
    imprint_linear_head_from_loader,
)


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


class DummyLVFaceViTBackbone(nn.Module):
    """模拟 third_party/LVFace VisionTransformer（顶层含 blocks / norm / feature）。"""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(24)])
        self.norm = nn.LayerNorm(4)
        self.feature = nn.Sequential(nn.Linear(4, 512))

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

    def test_cosface_subtracts_margin_on_target_class_only(self) -> None:
        head = CosFaceHead(in_features=2, num_classes=2, scale=10.0, margin=0.2)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        features = torch.tensor([[1.0, 0.0]])
        labels = torch.tensor([0])
        logits_train = head(features, labels)
        logits_infer = head(features, None)

        self.assertAlmostEqual(logits_train[0, 0].item(), logits_infer[0, 0].item() - 2.0, places=5)
        self.assertAlmostEqual(logits_train[0, 1].item(), logits_infer[0, 1].item(), places=5)

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

    @patch("dl_pipeline.models.classifier.load_lvface_backbone")
    def test_build_classifier_wraps_lvface_backbone(self, mock_lvface):
        mock_lvface.return_value = DummyLVFaceViTBackbone()

        model = build_classifier(
            model_family="lvface",
            backbone_name="vit_b_dp005_mask_005",
            num_classes=3,
            pretrained=False,
            dropout=0.2,
            pretrained_checkpoint_path="models/pretrained/LVFace-B_Glint360K.pt",
            freeze_backbone=True,
        )

        self.assertEqual(model.classifier[-1].out_features, 3)
        mock_lvface.assert_called_once()
        logits = model(torch.randn(2, 3, 112, 112))
        self.assertEqual(tuple(logits.shape), (2, 3))
        features = model.extract_features(torch.randn(2, 3, 112, 112))
        self.assertEqual(tuple(features.shape), (2, 512))

    def test_build_classifier_lvface_requires_pretrained_checkpoint_path(self):
        with self.assertRaises(ValueError):
            build_classifier(
                model_family="lvface",
                backbone_name="vit_b_dp005_mask_005",
                num_classes=3,
                pretrained=False,
                dropout=0.2,
                freeze_backbone=True,
            )

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

    @patch("dl_pipeline.models.classifier.resolve_pretrained_checkpoint")
    @patch("dl_pipeline.models.classifier.torch.load")
    def test_arcface_iresnet_last_stage_unfreezes_layer4_fc(self, mock_load, mock_resolve):
        from dl_pipeline.models.iresnet import iresnet50 as ir50

        backbone = ir50()
        mock_load.return_value = backbone.state_dict()
        mock_resolve.return_value = "dummy.pth"
        model = build_classifier(
            model_family="arcface_iresnet",
            backbone_name="r50",
            num_classes=3,
            pretrained=False,
            dropout=0.2,
            pretrained_checkpoint_path="dummy.pth",
            iresnet_finetune_mode="last_stage",
        )
        self.assertFalse(any(p.requires_grad for p in model.backbone.layer1.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.backbone.layer3.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.backbone.layer4.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.backbone.fc.parameters()))

    @patch("dl_pipeline.models.classifier.resolve_pretrained_checkpoint")
    @patch("dl_pipeline.models.classifier.torch.load")
    def test_arcface_iresnet_last_two_stages_unfreezes_layer3(self, mock_load, mock_resolve):
        from dl_pipeline.models.iresnet import iresnet50 as ir50

        backbone = ir50()
        mock_load.return_value = backbone.state_dict()
        mock_resolve.return_value = "dummy.pth"
        model = build_classifier(
            model_family="arcface_iresnet",
            backbone_name="r50",
            num_classes=3,
            pretrained=False,
            dropout=0.2,
            pretrained_checkpoint_path="dummy.pth",
            iresnet_finetune_mode="last_two_stages",
        )
        self.assertTrue(any(p.requires_grad for p in model.backbone.layer3.parameters()))

    @patch("dl_pipeline.models.classifier.resolve_pretrained_checkpoint")
    @patch("dl_pipeline.models.classifier.torch.load")
    def test_arcface_iresnet_invalid_finetune_mode_raises(self, mock_load, mock_resolve):
        from dl_pipeline.models.iresnet import iresnet50 as ir50

        mock_load.return_value = ir50().state_dict()
        mock_resolve.return_value = "dummy.pth"
        with self.assertRaises(ValueError):
            build_classifier(
                model_family="arcface_iresnet",
                backbone_name="r50",
                num_classes=3,
                pretrained=False,
                dropout=0.2,
                pretrained_checkpoint_path="dummy.pth",
                iresnet_finetune_mode="not_a_mode",
            )


class TinyImprintClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(4, 3))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class WeightImprintingTests(unittest.TestCase):
    def test_imprint_sets_unit_norm_class_prototypes(self) -> None:
        feats = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 6.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)
        loader = DataLoader(TensorDataset(feats, labels), batch_size=4)
        model = TinyImprintClassifier()
        imprint_linear_head_from_loader(model, loader, torch.device("cpu"), num_classes=3)
        linear = model.classifier[-1]
        w = linear.weight.data
        self.assertAlmostEqual(w[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(w[0, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(w[1, 1].item(), 1.0, places=5)
        # 类 2：两样本先 L2 归一化为 [0,0,1,0] 与 [0,0,0,1]，均值再 L2 归一化
        expected_2 = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        expected_2 = expected_2 / expected_2.norm(p=2)
        self.assertTrue(torch.allclose(w[2], expected_2, atol=1e-5))
        self.assertTrue(torch.allclose(linear.bias, torch.zeros(3)))


if __name__ == "__main__":
    unittest.main()
