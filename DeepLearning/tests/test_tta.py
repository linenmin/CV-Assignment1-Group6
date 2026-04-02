import unittest

import torch

from dl_pipeline.inference.tta import extract_tta_features


class DummyFeatureModel:
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        flattened = images.view(images.shape[0], -1)
        return flattened[:, :2]


class TTAInferenceTests(unittest.TestCase):
    def test_extract_tta_features_averages_original_and_horizontally_flipped_features(self):
        model = DummyFeatureModel()
        images = torch.tensor([[[[1.0, 0.0]]]])

        features = extract_tta_features(
            model=model,
            images=images,
            device=torch.device("cpu"),
            use_horizontal_flip=True,
        )

        expected = torch.tensor([[0.70710677, 0.70710677]])
        self.assertTrue(torch.allclose(features, expected, atol=1e-5))

    def test_extract_tta_features_without_flip_matches_original_features(self):
        model = DummyFeatureModel()
        images = torch.tensor([[[[3.0, 4.0]]]])

        features = extract_tta_features(
            model=model,
            images=images,
            device=torch.device("cpu"),
            use_horizontal_flip=False,
        )

        expected = torch.tensor([[0.6, 0.8]])
        self.assertTrue(torch.allclose(features, expected, atol=1e-5))
