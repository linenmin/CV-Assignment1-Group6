from __future__ import annotations

import torch

from dl_pipeline.inference.prototype import average_normalized_embedding_sets


def extract_tta_features(
    model,
    images: torch.Tensor,
    device: torch.device,
    use_horizontal_flip: bool = False,
) -> torch.Tensor:
    model_inputs = images.to(device)
    feature_sets = [model.extract_features(model_inputs).detach().cpu()]
    if use_horizontal_flip:
        flipped_inputs = torch.flip(model_inputs, dims=[3])
        feature_sets.append(model.extract_features(flipped_inputs).detach().cpu())
    return average_normalized_embedding_sets(feature_sets)
